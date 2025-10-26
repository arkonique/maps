# pop_sim_all_fast.py
import numpy as np
from scipy.ndimage import (
    distance_transform_edt,
    gaussian_filter,
    convolve,
    maximum_filter,
    label,  # C-optimized component labeling
)
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap, BoundaryNorm

# =========================================================
# Neighbor utilities
# =========================================================
NEIGH = np.array([
    (-1, -1), (-1, 0), (-1, 1),
    ( 0, -1),          ( 0, 1),
    ( 1, -1), ( 1, 0), ( 1, 1)
], dtype=np.int8)

KERNEL_8 = np.array([[1,1,1],[1,0,1],[1,1,1]], dtype=np.float32)

def roll2(a, dy, dx):
    # wraparound roll
    return np.roll(np.roll(a, dy, axis=0), dx, axis=1)

def one_step_route(amount, attractiveness, water_mask):
    """
    Move 'amount' one neighbor step toward argmax-attractiveness.
    Vectorized: stack 8 rolled attractiveness maps → argmax → scatter via rolls.
    """
    Astack = np.stack([roll2(attractiveness, -dy, -dx) for dy, dx in NEIGH], axis=0)  # [8,H,W]
    water_penalty = -1e20
    for k, (dy, dx) in enumerate(NEIGH):
        dest_is_water = roll2(water_mask, dy, dx)
        Astack[k][dest_is_water] = water_penalty
    best_dir = np.argmax(Astack, axis=0)
    inflow = np.zeros_like(amount, dtype=np.float32)
    for k, (dy, dx) in enumerate(NEIGH):
        take = amount * (best_dir == k)
        if np.any(take):
            inflow += roll2(take, dy, dx)
    return inflow

# =========================================================
# Biomes & weights
# IDs: 0=water, 1=sand, 2=grass, 3=forest, 4=rock, 5=snow
# =========================================================
def biome_id_from_scaled(land_scaled, water_mask, thresholds=(0.10, 0.40, 0.65, 0.85)):
    biome = np.zeros_like(land_scaled, dtype=np.uint8)
    biome[water_mask] = 0
    t1, t2, t3, t4 = thresholds
    s  = (~water_mask) & (land_scaled <  t1)
    g  = (~water_mask) & (land_scaled >= t1) & (land_scaled < t2)
    f  = (~water_mask) & (land_scaled >= t2) & (land_scaled < t3)
    r  = (~water_mask) & (land_scaled >= t3) & (land_scaled < t4)
    sn = (~water_mask) & (land_scaled >= t4)
    biome[s]  = 1; biome[g] = 2; biome[f] = 3; biome[r] = 4; biome[sn] = 5
    return biome

def biome_migration_weight(biome):
    # Migration attractiveness (destination preference).
    # Sand is ~neutral corridor (0.90), grass=1.00, forest slightly lower (0.85).
    return np.take(np.array([0.00, 0.90, 1.00, 0.85, 0.45, 0.10], np.float32), biome)

def biome_capacity_factor(biome):
    # Carrying capacity scaling; sand usable but lower than grass/forest.
    return np.take(np.array([0.00, 0.60, 1.00, 0.80, 0.40, 0.15], np.float32), biome)

def biome_death_multiplier(biome):
    # Mortality multiplier; sand only slightly worse than grass.
    return np.take(np.array([1.00, 1.02, 1.00, 1.05, 1.20, 1.40], np.float32), biome)

# =========================================================
# Conway-like Life (vectorized, probabilistic)
# =========================================================
def parse_life_rule(rule_str="B3/S23"):
    bpart, spart = rule_str.upper().split("/")
    B = set(int(ch) for ch in bpart[1:] if ch.isdigit())
    S = set(int(ch) for ch in spart[1:] if ch.isdigit())
    return B, S

def conway_prob_step(N, K_eff, rule_str="B3/S23",
                     pop_threshold=1200.0, density_norm=6000.0,
                     p_birth=0.9, p_survive=0.95, noise_sigma=0.05, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    B, S = parse_life_rule(rule_str)
    denom = pop_threshold + 0.25 * (K_eff + 1e-9)
    score = N / (denom + 1e-9)
    soft_alive = 1.0 / (1.0 + np.exp(-score))
    alive = soft_alive > (pop_threshold / (pop_threshold + density_norm))
    neigh = convolve(alive.astype(np.float32), KERNEL_8, mode="reflect")
    if noise_sigma > 0:
        pb = np.clip(p_birth   * np.exp(rng.normal(0.0, noise_sigma, size=N.shape)), 0.0, 1.0)
        ps = np.clip(p_survive * np.exp(rng.normal(0.0, noise_sigma, size=N.shape)), 0.0, 1.0)
    else:
        pb, ps = p_birth, p_survive
    dead = ~alive
    birth_mask   = dead  & np.isin(neigh, list(B))
    survive_mask = alive & np.isin(neigh, list(S))
    births_real  = birth_mask   & (rng.random(N.shape) < pb)
    survive_real = survive_mask & (rng.random(N.shape) < ps)
    alive_next = survive_real | births_real
    if len(B | S) > 0:
        ok = np.array(sorted(list(B | S)), dtype=np.float32)
        prox = np.min(np.stack([np.abs(neigh - k) for k in ok], axis=0), axis=0)
        targetish = 1.0 - (prox / 8.0)
    else:
        targetish = np.zeros_like(N, dtype=np.float32)
    L = np.clip(0.6 * soft_alive.astype(np.float32) + 0.4 * targetish, 0.0, 1.0)
    return alive_next, L

# =========================================================
# Heightmap utilities
# =========================================================
def heightmap_xyz_to_grid(heightmap_xyz, H=None, W=None, dtype=np.float32):
    """
    heightmap_xyz: array-like (N,3) with integer x,y and float z. If H,W not provided, infer from max x/y + 1.
    Returns h_raw[ y, x ] = z
    """
    xyz = np.asarray(heightmap_xyz)
    if H is None: H = int(xyz[:,1].max()) + 1
    if W is None: W = int(xyz[:,0].max()) + 1
    h_raw = np.zeros((H, W), dtype=dtype)
    x = xyz[:,0].astype(int); y = xyz[:,1].astype(int); z = xyz[:,2].astype(dtype)
    h_raw[y, x] = z
    return h_raw

# =========================================================
# Full simulation — fast & feature-complete (+ Exodus + Military)
# =========================================================
def simulate_population_from_heightmap(
    heightmap_xyz,
    # grid override (optional)
    H=None, W=None,
    # sea & biome mapping
    sea_level=0.0, Hmax=1.0, biome_thresholds=(0.10,0.40,0.65,0.85),
    # seeding / initialization
    initial_total_population=1_000_000, expected_settlements=250, spawn_gamma=1.3,
    # demography & capacity
    r_birth=0.022, r_death=0.014, K0=3000.0,
    kappa_water_capacity=1.0, lambda_water_capacity=8.0,  # capacity uplift decay from water
    # attractiveness (directed flow) — includes sand corridor & stronger water pull
    alpha_w=1.6, alpha_p=1.0, alpha_b=0.5, lambda_mig=8.0,
    water_affinity_gain=0.35, sand_corridor_boost=0.50, sand_mask_weight=None,
    density_sigma=1.8, rho_half=1800.0, pref_noise_sigma=0.12,
    # migration & flows (one-step each)
    m_dir=0.05,                      # directed (A_att)
    m_gravity=0.03, grav_sigma=7.0,  # gravity toward smoothed mass
    m_sprawl=0.02, urban_k_frac=0.60,# sprawl toward urban perimeter
    m_explore=0.004, m_diff=0.008,   # exploration + diffusion
    # infra & memory
    infra_sigma=2.0, infra_threshold=8000.0, infra_strength=0.20,
    memory_decay=0.98, memory_gain_scale=0.02, crowding_penalty=0.65,
    # Life coupling
    life_enabled=True, life_rule="B3/S23",
    life_pop_threshold=1200.0, life_density_norm=6000.0,
    life_p_birth=0.9, life_p_survive=0.95, life_noise_sigma=0.05,
    life_birth_boost=0.25, life_death_suppression=0.15,
    life_capacity_boost=0.20, life_migration_bias=0.25,
    # catastrophes (regional & persistent)
    catastrophe_prob=0.03, catastrophe_events_mean=1.0,
    region_shape_probs=(0.6, 0.4), disk_radius_range=(3, 8),
    box_hw_range=((5, 14), (5, 14)),
    catastrophe_types=None,
    # congestion
    delta_death=0.35, delta_forced_move=0.25,
    # churn (fast normal approx)
    lam_out=4e-4, lam_in=4e-4,
    # CULTURE factor (persistent towns)
    culture_enabled=True,
    town_threshold=2500.0, culture_radius=1, culture_persist_years=25,
    culture_min_abs=800.0, culture_min_frac=0.06,

    # ------------------ EXODUS ------------------
    exodus_enabled=True,
    exodus_prob=0.06, exodus_events_mean=1.0,
    exodus_min_pop=None, exodus_group_frac=0.25,
    exodus_min_dist=20, exodus_max_dist=None,
    exodus_target_bias_water=1.0, exodus_target_bias_empty=0.7,
    exodus_culture_factor=0.5,
    # ---------------------------------------------------

    # ------------------ MILITARY (sim-owned) ------------------
    military_enabled=True,                  # maintain per-cell stance in sim
    record_military=False,                  # store snapshots (aligned with save_every)
    military_init="random",                 # "random" init in [-1,1]
    military_forcing_enabled=False,         # periodic anti-neutral push
    military_neutral_band=0.08,             # |stance| <= band is "neutral"
    military_period=24,                     # frames/years per cycle
    military_strength=0.5,                  # max push per step (scaled by envelope & room)
    military_shape="sin",                   # "sin" | "saw" | "linear"
    # ---------------------------------------------------

    # run control
    years=300, seed=42, clip_negative=True, dtype=np.float32,
    save_every=1, return_meta=False, record_attractiveness=False,
):
    """
    Feature-complete, vectorized population simulation (heightmap-in).
    """
    rng = np.random.default_rng(seed)

    # -------- Terrain from heightmap --------
    h_raw = heightmap_xyz_to_grid(heightmap_xyz, H=H, W=W, dtype=dtype)
    H, W = h_raw.shape
    water_mask = h_raw <= sea_level
    land_mask  = ~water_mask

    land_scaled = np.zeros_like(h_raw, dtype=dtype)
    if np.any(land_mask):
        lo, hi = h_raw[land_mask].min(), h_raw[land_mask].max()
        denom = (hi - lo) if hi > lo else 1.0
        land_scaled[land_mask] = ((h_raw[land_mask] - lo) / denom) * Hmax

    biome = biome_id_from_scaled(land_scaled, water_mask, thresholds=biome_thresholds)
    D = distance_transform_edt(~water_mask.astype(bool)).astype(dtype)

    # -------- Static terrain fields --------
    W_bio_mig = biome_migration_weight(biome).astype(dtype)    # migration preference
    k_bio     = biome_capacity_factor(biome).astype(dtype)      # K factor
    death_mult = biome_death_multiplier(biome).astype(dtype)    # deaths
    sand_mask = (biome == 1)

    # Distance-to-water preference component for migration
    S_w = np.zeros((H, W), dtype=dtype)
    S_w[~water_mask] = np.exp(-D[~water_mask] / float(lambda_mig)).astype(dtype)

    # Carrying capacity uplift near water (separate decay)
    water_uplift = np.zeros((H, W), dtype=dtype)
    water_uplift[~water_mask] = np.exp(-D[~water_mask] / float(lambda_water_capacity)).astype(dtype)
    K_base = (K0 * k_bio * (1.0 + kappa_water_capacity * water_uplift)).astype(dtype)
    K_base[water_mask] = 0.0

    # -------- Sparse seeding --------
    pref = W_bio_mig * (1.0 + 0.5 * water_uplift)  # neutral sand + water proximity bias
    pref[water_mask] = 0.0
    probs = pref.ravel(); total = probs.sum()
    probs = (probs / total) if total > 0 else np.full(H*W, 1.0/(H*W), dtype=dtype)
    seeds = rng.choice(H*W, size=int(max(1, expected_settlements)), replace=False, p=probs)
    sy, sx = np.divmod(seeds, W)
    weights = np.clip(pref[sy, sx] ** spawn_gamma, 1e-9, None).astype(dtype)
    weights /= weights.sum()
    N = np.zeros((H, W), dtype=dtype)
    N[sy, sx] += (initial_total_population * weights).astype(dtype)

    # -------- State fields --------
    cum_mem = np.zeros((H, W), dtype=dtype)
    infra_field = np.zeros((H, W), dtype=dtype)

    # CULTURE state
    if culture_enabled:
        culture_counter = np.zeros((H, W), dtype=np.int32)
        culture_anchor  = np.zeros((H, W), dtype=bool)
    else:
        culture_counter = culture_anchor = None

    # ---- MILITARY per-cell state (sim-owned) ----
    if military_enabled:
        if military_init == "random":
            M_cell = rng.uniform(-1.0, 1.0, size=(H, W)).astype(np.float32)
        else:
            M_cell = rng.uniform(-1.0, 1.0, size=(H, W)).astype(np.float32)
        M_cell[water_mask] = 0.0
        M_runlen = np.zeros((H, W), dtype=np.int32)  # neutral streak length
        # deterministic fallback sign for cells where stance==0
        M_seed_sign = np.where(rng.integers(0, 2, size=(H, W), endpoint=False) == 0, -1, 1).astype(np.int8)
        M_seed_sign[water_mask] = 0
    else:
        M_cell = M_runlen = M_seed_sign = None

    # ---- histories ----
    pop_history = []
    att_history = [] if record_attractiveness else None
    mil_history = [] if (military_enabled and record_military) else None

    def maybe_store(t, arr_pop, arr_att=None, arr_mil=None):
        if t % save_every == 0:
            pop_history.append(arr_pop.copy())
            if record_attractiveness and (arr_att is not None):
                att_history.append(arr_att.copy())
            if mil_history is not None and (arr_mil is not None):
                mil_history.append(arr_mil.copy())

    # ---- compute baseline A_att for t=0 (no infra, no shocks, no life; memory=0) ----
    S_w_eff0 = np.clip(S_w * (1.0 + water_affinity_gain), 0.0, None) ** dtype(alpha_w)
    rho0     = gaussian_filter(np.zeros_like(N, dtype=np.float32), sigma=density_sigma, mode="reflect").astype(dtype)
    S_pop0   = rho0 / (rho0 + dtype(rho_half))
    crowd0   = np.clip(N / (K_base + 1e-9), 0.0, 2.0)
    crowd_t0 = np.exp(-crowding_penalty * crowd0).astype(dtype)
    mem_t0   = np.zeros_like(N, dtype=dtype)
    A_base0  = (S_w_eff0 * (S_pop0 ** dtype(alpha_p)) * (W_bio_mig ** dtype(alpha_b)) * crowd_t0 * (1.0 + 0.5 * mem_t0))
    A_sand_b0= (biome == 1).astype(dtype) * S_w_eff0 * dtype(sand_corridor_boost)
    A_att0   = A_base0 * (1.0 + A_sand_b0)

    # t=0 snapshots
    maybe_store(0, N, A_att0, (M_cell if military_enabled and record_military else None))

    # -------- Catastrophes defaults --------
    if catastrophe_types is None:
        catastrophe_types = {
            "war":     {"prob":0.40,"duration_range":(5,15),"mortality":0.04,"forced_migration":0.15,
                        "r_birth_mult":0.65,"r_death_add":0.006,"K_mult":0.92,"m_dir_mult":1.25},
            "famine":  {"prob":0.35,"duration_range":(3, 8),"mortality":0.02,"forced_migration":0.10,
                        "r_birth_mult":0.70,"r_death_add":0.008,"K_mult":0.65,"m_dir_mult":1.10},
            "epidemic":{"prob":0.15,"duration_range":(2, 5),"mortality":0.03,"forced_migration":0.05,
                        "r_birth_mult":0.60,"r_death_add":0.010,"K_mult":1.00,"m_dir_mult":0.85},
            "disaster":{"prob":0.10,"duration_range":(1, 3),"mortality":0.10,"forced_migration":0.20,
                        "r_birth_mult":1.00,"r_death_add":0.000,"K_mult":0.90,"m_dir_mult":1.30},
        }
    c_types = list(catastrophe_types.keys())
    c_probs = np.array([catastrophe_types[k]["prob"] for k in c_types], dtype=np.float64); c_probs /= c_probs.sum()
    active_shocks = []

    # Precompute coordinate grids for exodus distance checks
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")

    # ---- helpers (local) ----
    def sample_region_mask():
        flat = N.ravel(); tot = flat.sum()
        if tot <= 0:
            cy, cx = rng.integers(0, H), rng.integers(0, W)
        else:
            idx = rng.choice(H*W, p=(flat / tot)); cy, cx = divmod(idx, W)
        mask = np.zeros((H, W), dtype=bool)
        if rng.random() < region_shape_probs[0]:
            r = int(rng.integers(disk_radius_range[0], disk_radius_range[1] + 1))
            y0, x0 = cy, cx
            mask = (yy - y0)**2 + (xx - x0)**2 <= r**2
        else:
            (hmin,hmax),(wmin,wmax) = box_hw_range
            hh = int(rng.integers(hmin, hmax+1)); ww = int(rng.integers(wmin, wmax+1))
            y0 = np.clip(cy - hh//2, 0, H-1); x0 = np.clip(cx - ww//2, 0, W-1)
            y1 = np.clip(y0 + hh, 0, H);     x1 = np.clip(x0 + ww, 0, W)
            mask[y0:y1, x0:x1] = True
        mask &= (~water_mask)
        return mask

    def maybe_add_shocks():
        if rng.random() >= catastrophe_prob:
            return
        k = max(1, rng.poisson(catastrophe_events_mean))
        for _ in range(k):
            tname = rng.choice(c_types, p=c_probs); p = catastrophe_types[tname]
            dur = int(rng.integers(p["duration_range"][0], p["duration_range"][1] + 1))
            if dur <= 0: continue
            mask = sample_region_mask()
            if not mask.any(): continue
            active_shocks.append({
                "mask": mask, "years_left": dur,
                "mortality": float(p["mortality"]),
                "forced_migration": float(p["forced_migration"]),
                "r_birth_mult": float(p["r_birth_mult"]),
                "r_death_add": float(p["r_death_add"]),
                "K_mult": float(p["K_mult"]),
                "m_dir_mult": float(p["m_dir_mult"]),
            })

    def _military_envelope(runlen):
        if not military_forcing_enabled:
            return 0.0
        if military_period <= 1:
            phase = 1.0
        else:
            phase = (runlen % military_period).astype(np.float32) / float(military_period)
        if military_shape == "sin":
            env = np.sin(np.pi * phase)
        elif military_shape == "saw":
            env = phase
        else:
            env = np.where(phase <= 0.5, 2.0*phase, 2.0*(1.0 - phase)).astype(np.float32)
        return np.clip(env, 0.0, 1.0)

    # =========================
    # Yearly loop (vectorized)
    # =========================
    for t in range(1, years + 1):
        maybe_add_shocks()

        # Infra spillover
        hubs = (N >= infra_threshold) & (~water_mask)
        if hubs.any():
            infra_field = gaussian_filter(hubs.astype(dtype), sigma=infra_sigma, mode="reflect")
            m = infra_field.max(); infra_field = (infra_field / m) if m > 0 else infra_field*0
        else:
            infra_field *= 0

        # Shocks fields + annual shock mortality
        r_birth_mult_field = np.ones((H, W), dtype=dtype)
        r_death_add_field  = np.zeros((H, W), dtype=dtype)
        K_mult_field       = np.ones((H, W), dtype=dtype)
        m_dir_mult_field   = np.ones((H, W), dtype=dtype)
        if active_shocks:
            for sh in active_shocks:
                M = sh["mask"]
                mort = sh["mortality"]
                if mort > 0: N[M] *= (1.0 - mort)
                r_birth_mult_field[M] *= sh["r_birth_mult"]
                r_death_add_field[M]  += sh["r_death_add"]
                K_mult_field[M]       *= sh["K_mult"]
                m_dir_mult_field[M]   *= sh["m_dir_mult"]

        # Effective capacity
        K_eff = (K_base * (1.0 + infra_strength * infra_field) * K_mult_field).astype(dtype)
        K_eff[water_mask] = 0.0

        # Life coupling
        if life_enabled:
            _, L = conway_prob_step(
                N=N, K_eff=K_eff, rule_str=life_rule,
                pop_threshold=life_pop_threshold, density_norm=life_density_norm,
                p_birth=life_p_birth, p_survive=life_p_survive, noise_sigma=life_noise_sigma, rng=rng
            )
            K_eff = K_eff * (1.0 + life_capacity_boost * L)
        else:
            L = np.zeros_like(N, dtype=dtype)

        # Demography (logistic)
        r_birth_eff = r_birth * r_birth_mult_field * (1.0 + life_birth_boost * L)
        r_death_eff = (r_death + r_death_add_field) * death_mult * (1.0 - life_death_suppression * L)
        r_net_local = r_birth_eff - r_death_eff
        growth = r_net_local * N * (1.0 - np.divide(N, K_eff, out=np.zeros_like(N), where=(K_eff > 0)))
        N_prime = N + growth
        N_prime[water_mask] = 0

        # Memory update
        cum_mem = memory_decay * cum_mem + memory_gain_scale * N_prime

        # ---------- Attractiveness for standard directed flow ----------
        S_w_eff = np.clip(S_w * (1.0 + water_affinity_gain), 0.0, None) ** dtype(alpha_w)
        rho   = gaussian_filter(N.astype(np.float32), sigma=density_sigma, mode="reflect").astype(dtype)
        S_pop = rho / (rho + dtype(rho_half))
        crowd = np.clip(N / (K_eff + 1e-9), 0.0, 2.0)
        crowd_term = np.exp(-crowding_penalty * crowd).astype(dtype)
        mem_term   = np.clip(cum_mem / (cum_mem.max() + 1e-9), 0, 1).astype(dtype) ** dtype(0.8)

        A_base = (S_w_eff * (S_pop ** dtype(alpha_p)) * (W_bio_mig ** dtype(alpha_b)) * crowd_term * (1.0 + 0.5 * mem_term))
        A_sand_bonus = sand_mask.astype(dtype) * S_w_eff * dtype(sand_corridor_boost)
        A_att = A_base * (1.0 + A_sand_bonus)
        if life_enabled and life_migration_bias != 0:
            A_att = A_att * (1.0 + life_migration_bias * L)
        if pref_noise_sigma > 0:
            A_att = A_att * np.exp(rng.normal(0.0, pref_noise_sigma, size=A_att.shape)).astype(dtype)

        # -------- Flows (one-step, vectorized) --------
        # (a) Directed flow
        M_dir_out = (m_dir * m_dir_mult_field) * N_prime
        N_after = N_prime - M_dir_out
        inflow_dir = one_step_route(M_dir_out, A_att, water_mask)

        # (b) Gravity flow — toward smoothed mass field
        if m_gravity > 0:
            Pgrav = gaussian_filter(N_after.astype(np.float32), sigma=grav_sigma, mode="reflect").astype(dtype)
            M_grav_out = m_gravity * N_after
            N_after -= M_grav_out
            inflow_grav = one_step_route(M_grav_out, Pgrav, water_mask)
        else:
            inflow_grav = 0.0

        # (c) Urban sprawl — to urban perimeter
        if m_sprawl > 0:
            U = (N >= urban_k_frac * K_eff) & (~water_mask)
            U_neighbors = convolve(U.astype(np.float32), KERNEL_8, mode="reflect")
            A_sprawl = U_neighbors * (~U)  # attractive just outside cities
            M_sprawl_out = m_sprawl * N_after
            N_after -= M_sprawl_out
            inflow_sprawl = one_step_route(M_sprawl_out, A_sprawl, water_mask)
        else:
            inflow_sprawl = 0.0

        # (d) Exploration (small random neighbor diffusion)
        if m_explore > 0:
            avg_nbr = convolve(N_after, KERNEL_8, mode="wrap") / 8.0
            N_after = N_after + m_explore * (avg_nbr - N_after)

        # (e) Diffusion (isotropic)
        if m_diff > 0:
            avg_nbr2 = convolve(N_after, KERNEL_8, mode="wrap") / 8.0
            N_after = N_after + m_diff * (avg_nbr2 - N_after)

        # Aggregate flows
        N_temp = N_after + inflow_dir + (inflow_grav if isinstance(inflow_grav, np.ndarray) else 0.0) + (inflow_sprawl if isinstance(inflow_sprawl, np.ndarray) else 0.0)
        N_temp[water_mask] = 0

        # Congestion: extra deaths + forced one-step out along A_att
        over = np.maximum(0.0, N_temp - K_eff)
        extra_death = delta_death * over
        forced_out = delta_forced_move * over
        N_temp = N_temp - (extra_death + forced_out)
        N_temp[N_temp < 0] = 0
        N_temp += one_step_route(forced_out, A_att, water_mask)
        N_temp[water_mask] = 0

        # Persistent shock-driven outflow (regional masks)
        if active_shocks:
            total_forced = np.zeros_like(N_temp, dtype=dtype)
            for sh in active_shocks:
                if sh["forced_migration"] <= 0: continue
                total_forced += sh["forced_migration"] * (sh["mask"].astype(dtype)) * N_temp
            N_temp -= total_forced
            N_temp += one_step_route(total_forced, A_att, water_mask)
            N_temp[water_mask] = 0

        # Exogenous churn (normal approx)
        if lam_in > 0 or lam_out > 0:
            mean_out = lam_out * N_temp
            mean_in  = lam_in  * N_temp
            Xi_out = np.maximum(0.0, rng.normal(mean_out, np.sqrt(mean_out + 1e-9), size=N_temp.shape)).astype(dtype)
            Xi_in  = np.maximum(0.0, rng.normal(mean_in,  np.sqrt(mean_in  + 1e-9), size=N_temp.shape)).astype(dtype)
            N_next = N_temp - Xi_out + Xi_in
        else:
            N_next = N_temp

        if clip_negative: N_next[N_next < 0] = 0
        N_next[water_mask] = 0

        # ------------------ EXODUS ------------------
        if exodus_enabled and rng.random() < exodus_prob:
            k_events = max(1, rng.poisson(exodus_events_mean))
            min_pop = float(town_threshold if exodus_min_pop is None else exodus_min_pop)
            # bias field from current N_next
            rho_next = gaussian_filter(N_next.astype(np.float32), sigma=density_sigma, mode="reflect").astype(dtype)
            S_pop_next = rho_next / (rho_next + dtype(rho_half))

            # candidate sources
            cand = (N_next >= min_pop) & (~water_mask)
            if np.any(cand):
                src_probs = (N_next * cand).ravel()
                src_probs_sum = src_probs.sum()
                if src_probs_sum > 0:
                    src_probs = src_probs / src_probs_sum
                    for _ in range(k_events):
                        sidx = rng.choice(H*W, p=src_probs)
                        sy, sx = divmod(sidx, W)
                        src_pop = float(N_next[sy, sx])
                        if src_pop < min_pop:
                            continue
                        group = max(1.0, exodus_group_frac * src_pop)
                        N_next[sy, sx] = max(0.0, src_pop - group)

                        dist = np.maximum(np.abs(yy - sy), np.abs(xx - sx))
                        target_mask = (~water_mask) & (dist >= int(exodus_min_dist))
                        if exodus_max_dist is not None:
                            target_mask &= (dist <= int(exodus_max_dist))

                        if exodus_target_bias_water > 0 or exodus_target_bias_empty > 0:
                            w = np.ones((H, W), dtype=np.float64)
                            if exodus_target_bias_water > 0:
                                w *= (1e-6 + S_w)
                            if exodus_target_bias_empty > 0:
                                w *= (1e-6 + (1.0 - S_pop_next))
                        else:
                            w = np.ones((H, W), dtype=np.float64)

                        w *= target_mask.astype(np.float64)
                        wsum = w.sum()
                        if wsum <= 0:
                            fallback = np.where(target_mask.ravel())[0]
                            if fallback.size == 0:
                                N_next[sy, sx] += group
                                continue
                            tidx = int(rng.choice(fallback))
                        else:
                            w = (w / wsum).ravel()
                            tidx = int(rng.choice(H*W, p=w))

                        ty, tx = divmod(tidx, W)
                        N_next[ty, tx] += group

                        if culture_enabled:
                            boost = int(max(0, round(exodus_culture_factor * culture_persist_years)))
                            culture_counter[ty, tx] = max(culture_counter[ty, tx], boost)

        # ===== CULTURE: update & enforce floor =====
        if culture_enabled:
            size = 2 * int(culture_radius) + 1
            local_max = maximum_filter(N, size=size, mode="reflect")
            stable = local_max >= float(town_threshold)
            culture_counter = np.where(stable, culture_counter + 1, 0)
            culture_anchor = culture_anchor | (culture_counter >= int(culture_persist_years))
            floor = np.maximum(culture_min_abs, culture_min_frac * K_eff)
            N_next = np.where(culture_anchor & (~water_mask), np.maximum(N_next, floor), N_next)

        # ---- MILITARY periodic forcing (sim-time) ----
        if military_enabled and military_forcing_enabled:
            neutral = (np.abs(M_cell) <= float(military_neutral_band))
            M_runlen = np.where(neutral, M_runlen + 1, 0)
            env = _military_envelope(M_runlen).astype(np.float32)
            sgn = np.sign(M_cell).astype(np.int8)
            sgn = np.where(sgn == 0, M_seed_sign, sgn).astype(np.int8)
            room = (1.0 - np.clip(np.abs(M_cell), 0.0, 1.0)).astype(np.float32)
            delta = (military_strength * env * room).astype(np.float32)
            M_cell = np.clip(M_cell + (sgn.astype(np.float32) * delta), -1.0, 1.0)
            M_cell[water_mask] = 0.0

        # Expire shocks
        if active_shocks:
            for sh in active_shocks: sh["years_left"] -= 1
            active_shocks = [sh for sh in active_shocks if sh["years_left"] > 0]

        # advance state
        N = N_next.astype(dtype, copy=False)

        # store snapshots
        maybe_store(t, N, A_att, (M_cell if (mil_history is not None) else None))

    # <<< END OF LOOP >>>
    # expose last military state (even if not recording history)
    military_trait = (M_cell.astype(np.float32) if military_enabled else None)

    if return_meta:
        meta = dict(
            biome=biome, water_mask=water_mask, D=D, land_scaled=land_scaled, h_raw=h_raw,
            culture_anchor=(culture_anchor if culture_enabled else None),
            K_base=K_base,
            military_trait=military_trait,  # last per-cell stance
        )
        # return structure:
        # - if record_attractiveness and record_military: ((pops, atts), mils), meta
        # - if only one: (pops, that_one), meta
        if record_attractiveness and (att_history is not None) and (mil_history is not None):
            return ((pop_history, att_history), mil_history), meta
        elif record_attractiveness and (att_history is not None):
            return (pop_history, att_history), meta
        elif mil_history is not None:
            return (pop_history, mil_history), meta
        else:
            return pop_history, meta

    if record_attractiveness and (att_history is not None):
        return (pop_history, att_history)
    if mil_history is not None:
        return (pop_history, mil_history)
    return pop_history

# =========================================================
# Visualizer (biome base + population/attractiveness/military + totals)
# =========================================================
def visualize_population_with_totals(
    pop_history,
    meta=None,
    # ---- view mode knob ----
    view="biome_overlay",            # "biome_overlay" | "population" | "attractiveness" | "military"

    # ---- overlay knobs (biome_overlay) ----
    overlay_metric="population",     # "population" or "military"
    highlight_frac=0.50,
    highlight_basis="start_max",     # "start_max" | "global_max" | "frame_max"
    # alpha controls (apply to both population & military overlays)
    overlay_alpha_gamma=0.6,         # higher = more selective (sharper fade-in)
    overlay_alpha_min=0.0,           # floor alpha
    overlay_alpha_max=1.0,           # cap alpha
    # colormaps
    overlay_cmap="magma",            # for population overlay
    overlay_military_cmap="coolwarm",
    # military overlay options
    overlay_cluster_threshold=2500.0,
    overlay_use_culture=False,
    overlay_military_color_window=(-0.5, 0.5),  # force color scale window (min,max)

    # ---- population-only knobs ----
    pop_cmap="magma",
    pop_log=False,
    pop_vmax=None,
    pop_vmax_mode="global",          # "global" | "frame"

    # ---- attractiveness view knobs ----
    att_history=None,                # REQUIRED when view="attractiveness"
    att_cmap="viridis",
    att_log=False,
    att_vmax=None,
    att_vmax_mode="global",

    # ---- military full-view knobs ----
    military_cmap="coolwarm",
    military_color_window=(-0.5, 0.5),
    cluster_threshold=2500.0,
    cluster_use_culture=False,

    # ---- inputs from sim (preferred for military) ----
    military_cell_history=None,      # list/array of [frames][H,W] in [-1,1]

    # ---- COUNTRY BOUNDARIES (based on |stance|) ----
    country_overlay=False,           # draw borders derived from military power
    country_base_radius=2.0,         # base expansion in cells
    country_scale=12.0,              # extra expansion per |stance|^gamma
    country_gamma=1.0,               # nonlinearity on |stance|
    country_min_stance=0.05,         # ignore near-neutral clusters
    country_edge_alpha=0.95,
    country_edge_color=None,         # None -> auto black/white for contrast

    # ---- animation & layout ----
    fps=12,
    title_prefix="Population — Year ",
    save_path=None,
    show=True,
    first_year=0,
    step=1,
    last_frame_png_path=None,
):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from scipy.ndimage import label, distance_transform_edt

    frames = len(pop_history)
    H, W = pop_history[0].shape
    totals = np.array([f.sum() for f in pop_history], dtype=np.float64)

    start_max = float(np.max(pop_history[0]))
    global_max = float(max(np.max(f) for f in pop_history))

    fig = plt.figure(figsize=(9, 7))
    gs  = fig.add_gridspec(3, 1, height_ratios=[3.0, 0.15, 1.0])
    ax_top = fig.add_subplot(gs[0, 0])
    ax_bot = fig.add_subplot(gs[2, 0])
    ttl = ax_top.set_title(f"{title_prefix}{first_year}")
    ax_top.set_xticks([]); ax_top.set_yticks([])

    def to_log(arr, flag):
        return np.log1p(arr) if flag else arr

    # ---------- helpers ----------
    def _stance_map(pop, military_cell, water_mask, thr, use_culture, culture_anchor=None):
        # settlement mask
        if use_culture and (culture_anchor is not None):
            mask = (culture_anchor.astype(bool)) & (~water_mask)
        else:
            mask = (pop >= float(thr)) & (~water_mask)

        lbl, nlab = label(mask.astype(np.int8), structure=np.array([[1,1,1],[1,1,1],[1,1,1]], dtype=np.int8))
        if nlab == 0:
            out = np.zeros_like(pop, dtype=np.float32)
            out[water_mask] = 0.0
            return out, lbl, nlab, mask

        lab_flat = lbl.ravel()
        valid = lab_flat > 0
        idx = lab_flat[valid].astype(np.int64)
        pop_flat = pop.ravel()[valid].astype(np.float64)
        m_flat   = military_cell.ravel()[valid].astype(np.float64)

        sum_pop = np.bincount(idx, weights=pop_flat, minlength=nlab + 1)
        sum_m   = np.bincount(idx, weights=pop_flat * m_flat, minlength=nlab + 1)
        with np.errstate(divide="ignore", invalid="ignore"):
            means = sum_m[1:] / np.maximum(sum_pop[1:], 1e-12)
        means = np.clip(means, -1.0, 1.0).astype(np.float32)

        means_pad = np.zeros(nlab + 1, dtype=np.float32); means_pad[1:] = means
        stance_map = means_pad[lbl]
        stance_map[water_mask] = 0.0
        return stance_map, lbl, nlab, mask

    def _alpha_from_strength(strength):
        strength = np.clip(strength, 0.0, 1.0)
        return overlay_alpha_min + (overlay_alpha_max - overlay_alpha_min) * (strength ** overlay_alpha_gamma)

    def _apply_color_window(values, window):
        vmin, vmax = window
        if vmax <= vmin: vmax = vmin + 1e-6
        normed = (values - vmin) / (vmax - vmin)
        return np.clip(normed, 0.0, 1.0)

    def _military_cell(i):
        if military_cell_history is not None:
            return military_cell_history[i].astype(np.float32)
        if (meta is not None) and ("military_trait" in meta) and (meta["military_trait"] is not None):
            return meta["military_trait"].astype(np.float32)
        return np.zeros_like(pop_history[0], dtype=np.float32)

    def _country_labels(stance_map, lbl, nlab, mask, water_mask):
        """
        Expand each labeled settlement into a 'country' with radius depending on |stance|.
        Returns: country_lbl (H,W) where 0=none, k>=1 is country id.
        Clipped to land (~water_mask).
        """
        if nlab == 0:
            return np.zeros_like(lbl, dtype=np.int32)

        # per-label |stance| mean
        lab_flat = lbl.ravel()
        valid = lab_flat > 0
        idx = lab_flat[valid].astype(np.int64)
        s_flat = stance_map.ravel()[valid].astype(np.float64)
        sum_abs = np.bincount(idx, weights=np.abs(s_flat), minlength=nlab + 1)
        cnt     = np.bincount(idx, minlength=nlab + 1).astype(np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            mean_abs = sum_abs[1:] / np.maximum(cnt[1:], 1e-12)
        mean_abs = np.nan_to_num(mean_abs, nan=0.0).astype(np.float32)
        mean_abs[mean_abs < float(country_min_stance)] = 0.0

        # radii per label
        radii = country_base_radius + country_scale * (mean_abs ** country_gamma)
        radii = np.maximum(0.0, radii).astype(np.float32)

        # distance to nearest settlement seed (mask)
        inv = ~mask
        dist, ind = distance_transform_edt(inv, return_indices=True)
        ny, nx = ind[0], ind[1]
        nearest_lbl = lbl[ny, nx]

        # per-pixel radius of its nearest label
        radii_pad = np.zeros(nlab + 1, dtype=np.float32)
        radii_pad[1:] = radii
        R = radii_pad[nearest_lbl]

        # inside country if (nearest label exists) & (within radius) & (on land)
        inside = (nearest_lbl > 0) & (dist <= R) & (~water_mask)
        country_lbl = np.where(inside, nearest_lbl, 0).astype(np.int32)
        return country_lbl


    culture_anchor = meta.get("culture_anchor") if meta is not None else None
    water_mask = meta.get("water_mask") if meta is not None else np.zeros_like(pop_history[0], dtype=bool)

    # =========================
    # VIEW ROUTING
    # =========================
    if view == "biome_overlay":
        if meta is None or "biome" not in meta:
            raise ValueError("meta with 'biome' is required for biome-colored base when view='biome_overlay'.")
        biome = meta["biome"].astype(np.int16)

        # base biome tiles
        biome_colors = np.array([
            [ 54,  90, 154],  # water
            [237, 201, 175],  # sand
            [120, 182,  96],  # grass
            [ 52, 120,  73],  # forest
            [130, 130, 130],  # rock
            [240, 245, 250],  # snow
        ], dtype=np.float32) / 255.0
        biome_cmap = ListedColormap(biome_colors)
        biome_norm = BoundaryNorm(boundaries=[-0.5,0.5,1.5,2.5,3.5,4.5,5.5], ncolors=biome_cmap.N)
        base_img = ax_top.imshow(biome, cmap=biome_cmap, norm=biome_norm, origin="upper")

        if overlay_metric not in {"population", "military"}:
            raise ValueError("overlay_metric must be 'population' or 'military'.")

        # Precompute global basis if needed
        if overlay_metric == "population":
            basis_global = float(max(np.max(f) for f in pop_history)) if highlight_basis == "global_max" else None
        else:
            def _stance_abs_max(i):
                pop = pop_history[i].astype(np.float32)
                mcell = _military_cell(i)
                stance, lbl, nlab, mask = _stance_map(pop, mcell, water_mask,
                                                      overlay_cluster_threshold, overlay_use_culture, culture_anchor)
                return float(np.max(np.abs(stance)))
            basis_global = None
            if highlight_basis == "global_max":
                basis_global = 0.0
                for i in range(frames):
                    basis_global = max(basis_global, _stance_abs_max(i))

        def make_overlay(i):
            pop = pop_history[i].astype(np.float32)
            if overlay_metric == "population":
                basis = {
                    "start_max": float(np.max(pop_history[0])),
                    "global_max": (basis_global if basis_global is not None else float(np.max(pop))),
                    "frame_max": float(np.max(pop)),
                }[highlight_basis]
                thr = highlight_frac * (basis if basis > 0 else 1.0)
                rngv = max(1e-9, (basis - thr))
                strength = np.clip((pop - thr) / rngv, 0.0, 1.0)  # [0,1]
                alpha = _alpha_from_strength(strength)
                cmap = plt.get_cmap(overlay_cmap)
                rgba = cmap(np.clip(pop / max(1e-9, basis), 0, 1))

                # borders off in population overlay
                rgba[..., 3] = alpha
                return rgba
            else:
                mcell = _military_cell(i)
                stance, lbl, nlab, mask = _stance_map(pop, mcell, water_mask,
                                                      overlay_cluster_threshold, overlay_use_culture, culture_anchor)

                # basis for alpha (abs stance)
                if highlight_basis == "start_max":
                    pop0 = pop_history[0].astype(np.float32)
                    m0 = _military_cell(0)
                    st0, _, _, _ = _stance_map(pop0, m0, water_mask,
                                               overlay_cluster_threshold, overlay_use_culture, culture_anchor)
                    basis = float(np.max(np.abs(st0)))
                elif highlight_basis == "global_max":
                    basis = basis_global if basis_global is not None else float(np.max(np.abs(stance)))
                else:
                    basis = float(np.max(np.abs(stance)))
                basis = basis if basis > 0 else 1.0

                thr = highlight_frac * basis
                rngv = max(1e-9, (basis - thr))
                strength = np.clip((np.abs(stance) - thr) / rngv, 0.0, 1.0)
                alpha = _alpha_from_strength(strength)

                cmap = plt.get_cmap(overlay_military_cmap)
                normed = _apply_color_window(stance, overlay_military_color_window)
                rgba = cmap(normed)
                rgba[..., 3] = alpha

                # ---- Country borders (optional) ----
                # ---- Country borders (optional) ----
                if country_overlay:
                    country_lbl = _country_labels(stance, lbl, nlab, mask, water_mask)
                    L = country_lbl
                    land = ~water_mask
                    nz = (L > 0) & land

                    # borders between different countries OR next to water
                    neighbor_diff = (
                        (L != np.roll(L, 1, axis=0)) |
                        (L != np.roll(L,-1, axis=0)) |
                        (L != np.roll(L, 1, axis=1)) |
                        (L != np.roll(L,-1, axis=1))
                    )
                    coast_touch = (
                        np.roll(water_mask, 1, axis=0) |
                        np.roll(water_mask,-1, axis=0) |
                        np.roll(water_mask, 1, axis=1) |
                        np.roll(water_mask,-1, axis=1)
                    )
                    edge = nz & (neighbor_diff | coast_touch)

                    # paint edges solid black
                    rgba[edge, :3] = 0.0
                    rgba[edge,  3] = country_edge_alpha


                return rgba

        im_top = ax_top.imshow(make_overlay(0), origin="upper")

        def update_top(i):
            im_top.set_data(make_overlay(i))
            return (im_top,)

        update_top(0)

    elif view == "population":
        pop0 = to_log(pop_history[0].astype(np.float32), pop_log)
        vmax0 = pop_vmax if pop_vmax is not None else (
            float(max(np.max(to_log(f.astype(np.float32), pop_log)) for f in pop_history))
            if pop_vmax_mode == "global" else float(pop0.max())
        )
        im_top = ax_top.imshow(pop0, cmap=pop_cmap, vmin=0.0, vmax=max(1e-9, vmax0), origin="upper")
        cb = plt.colorbar(im_top, ax=ax_top, fraction=0.046, pad=0.02)
        cb.set_label("log1p(population)" if pop_log else "population per cell")

        def update_top(i):
            fld = to_log(pop_history[i].astype(np.float32), pop_log)
            if pop_vmax is None and pop_vmax_mode == "frame":
                im_top.set_clim(0.0, max(1e-9, float(fld.max())))
            im_top.set_data(fld)
            return (im_top,)

        update_top(0)

    elif view == "attractiveness":
        if att_history is None:
            raise ValueError("att_history is required when view='attractiveness'.")
        att0 = to_log(att_history[0].astype(np.float32), att_log)
        vmax0 = att_vmax if att_vmax is not None else (
            float(max(np.max(to_log(a.astype(np.float32), att_log)) for a in att_history))
            if att_vmax_mode == "global" else float(att0.max())
        )
        im_top = ax_top.imshow(att0, cmap=att_cmap, vmin=0.0, vmax=max(1e-9, vmax0), origin="upper")
        cb = plt.colorbar(im_top, ax=ax_top, fraction=0.046, pad=0.02)
        cb.set_label("log1p(attractiveness)" if att_log else "attractiveness")

        def update_top(i):
            fld = to_log(att_history[i].astype(np.float32), att_log)
            if att_vmax is None and att_vmax_mode == "frame":
                im_top.set_clim(0.0, max(1e-9, float(fld.max())))
            im_top.set_data(fld)
            return (im_top,)

        update_top(0)

    elif view == "military":
        if meta is None or water_mask is None:
            raise ValueError("meta with 'water_mask' is required for military view.")

        def stance_frame(i):
            pop = pop_history[i].astype(np.float32)
            mcell = _military_cell(i)
            st, lbl, nlab, mask = _stance_map(pop, mcell, water_mask,
                                              cluster_threshold, cluster_use_culture, culture_anchor)
            if country_overlay:
                country_lbl = _country_labels(st, lbl, nlab, mask)
                return st, country_lbl
            return st, None

        st0, c0 = stance_frame(0)
        normed0 = _apply_color_window(st0, military_color_window)
        im_top = ax_top.imshow(normed0, cmap=military_cmap, vmin=0.0, vmax=1.0, origin="upper")
        cb = plt.colorbar(im_top, ax=ax_top, fraction=0.046, pad=0.02)
        cb.set_label(f"military stance mapped via window {military_color_window}")

        # add a second image for borders if needed
        if country_overlay:
            border_img = ax_top.imshow(np.zeros((H, W, 4), dtype=np.float32), origin="upper")

        def _edge_rgba(country_lbl):
            if country_lbl is None:
                return None
            L = country_lbl
            nz = L > 0
            edge = nz & (
                (L != np.roll(L, 1, axis=0)) |
                (L != np.roll(L,-1, axis=0)) |
                (L != np.roll(L, 1, axis=1)) |
                (L != np.roll(L,-1, axis=1))
            )
            rgba = np.zeros((H, W, 4), dtype=np.float32)
            if country_edge_color is None:
                base = im_top.get_array()
                # approximate brightness from current colormap output (already 0..1)
                # If it’s a scalar image, map through the cmap; here we pick black/white default:
                edge_rgb = np.zeros((H, W, 3), dtype=np.float32)
            else:
                from matplotlib.colors import to_rgb
                c = np.array(to_rgb(country_edge_color), dtype=np.float32)
                edge_rgb = np.broadcast_to(c, (H, W, 3)).copy()
            rgba[edge, :3] = edge_rgb[edge]
            rgba[edge, 3]  = country_edge_alpha
            return rgba

        if country_overlay and c0 is not None:
            border_img.set_data(_edge_rgba(c0))

        def update_top(i):
            st, cL = stance_frame(i)
            im_top.set_data(_apply_color_window(st, military_color_window))
            if country_overlay and cL is not None:
                border_img.set_data(_edge_rgba(cL))
                return (im_top, border_img)
            return (im_top,)

        update_top(0)

    else:
        raise ValueError("view must be 'biome_overlay' or 'population' or 'attractiveness' or 'military'.")

    # timeline
    x = np.arange(frames) * step + first_year
    (line,) = ax_bot.plot(x[:1], totals[:1])
    ax_bot.set_xlim(x[0], x[-1])
    ax_bot.set_ylim(0, totals.max() * 1.05 if totals.max() > 0 else 1.0)
    ax_bot.set_xlabel("Year"); ax_bot.set_ylabel("Total population")
    vline = ax_bot.axvline(x[0], linestyle="--", alpha=0.7)

    def update(i):
        artists = update_top(i)
        ttl.set_text(f"{title_prefix}{x[i]}")
        line.set_data(x[:i+1], totals[:i+1])
        vline.set_xdata([x[i], x[i]])
        return (*artists, ttl, line, vline)

    anim = FuncAnimation(fig, update, frames=frames, interval=1000/max(1, fps), blit=False)

    if last_frame_png_path is not None:
        update(frames - 1)
        fig.canvas.draw_idle()
        fig.savefig(last_frame_png_path, dpi=220, bbox_inches="tight")

    if show:
        plt.show()
    if save_path is not None:
        anim.save(save_path, dpi=140)

    plt.close(fig)
    return anim

# =========================================================
# Demo (you can replace the heightmap with your own)
# =========================================================
if __name__ == "__main__":
    H, W = 96, 128
    rng = np.random.default_rng(3)
    z = gaussian_filter(rng.standard_normal((H, W)).astype(np.float32), sigma=12.0, mode="reflect")
    z = (z - z.mean()) / (z.std() + 1e-6)
    xs, ys = np.meshgrid(np.arange(W), np.arange(H))
    heightmap_xyz = np.column_stack([xs.ravel(), ys.ravel(), z.ravel()])

    # --- run sim; record both attractiveness and military histories
    ((pops, atts), mils), meta = simulate_population_from_heightmap(
        heightmap_xyz,
        sea_level=0.0,
        years=800, seed=11,
        initial_total_population=1_600_000, expected_settlements=180,
        # flows
        m_dir=0.05,
        m_gravity=0.035, grav_sigma=7.0,
        m_sprawl=0.022, urban_k_frac=0.58,
        m_explore=0.004, m_diff=0.009,
        # infra & robustness
        infra_sigma=2.0, infra_threshold=9000.0, infra_strength=0.25,
        pref_noise_sigma=0.10, crowding_penalty=0.70,
        # Life
        life_enabled=True, life_rule="B3/S23",
        life_birth_boost=0.26, life_death_suppression=0.15, life_capacity_boost=0.22, life_migration_bias=0.22,
        # Catastrophes
        catastrophe_prob=0.03, catastrophe_events_mean=1.1, disk_radius_range=(3,7),
        # Culture
        culture_enabled=True, town_threshold=2600.0, culture_radius=1, culture_persist_years=20,
        culture_min_abs=900.0, culture_min_frac=0.06,
        # Sand corridor & stronger water pull
        alpha_w=1.6, water_affinity_gain=0.35, sand_corridor_boost=0.6,
        # Exodus
        exodus_enabled=True, exodus_prob=0.07, exodus_events_mean=1.2,
        exodus_group_frac=0.22, exodus_min_dist=50,
        exodus_max_dist=None, exodus_target_bias_water=0.0,
        exodus_target_bias_empty=0.8, exodus_culture_factor=0.5,
        # Military forcing in the sim
        military_enabled=True,
        record_military=True,              # <-- store per-frame military arrays
        military_forcing_enabled=True,     # <-- run periodic anti-neutral push
        military_neutral_band=0.10,
        military_period=24,
        military_strength=0.6,
        military_shape="sin",
        # recording & performance
        save_every=1, dtype=np.float32, return_meta=True, record_attractiveness=True
    )
    print("Frames:", len(pops), "Grid:", pops[0].shape, "Start:", int(pops[0].sum()), "End:", int(pops[-1].sum()))

    # 1) Biome + population overlay with last-frame png
    visualize_population_with_totals(
        pops, meta,
        view="biome_overlay",
        overlay_metric="population",
        highlight_frac=0.1,
        fps=12,
        last_frame_png_path="population_last.png"
    )

    # 2) Biome + military overlay (alpha by |stance| vs threshold; colors via window)
    visualize_population_with_totals(
        pops, meta,
        view="biome_overlay",
        overlay_metric="military",
        overlay_military_color_window=(-0.5, 0.5),
        overlay_alpha_gamma=1.1,
        overlay_alpha_min=0.6,
        overlay_alpha_max=1,
        overlay_cluster_threshold=2600.0,
        military_cell_history=mils,     # from the sim
        # countries:
        country_overlay=True,
        country_base_radius=2.0,
        country_scale=14.0,
        country_gamma=1.2,
        country_min_stance=0.08,
        country_edge_alpha=0.95,
        # country_edge_color="black",   # optional override
        title_prefix="Military + Countries — Year ",
        fps=12,
    )


    # 3) Full military view (time-evolving)
    visualize_population_with_totals(
        pop_history=pops,
        meta=meta,
        view="military",
        military_cell_history=mils,         # <-- from sim
        military_cmap="coolwarm",
        military_color_window=(-0.5, 0.5),
        cluster_threshold=2600.0,
        cluster_use_culture=False,
        fps=12,
        title_prefix="Military — Year ",
        last_frame_png_path="military_last.png"
    )
