# pop_sim_all_fast.py
import numpy as np
from scipy.ndimage import (
    distance_transform_edt,
    gaussian_filter,
    convolve,
    maximum_filter,
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
# Full simulation — fast & feature-complete (+ Exodus)
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
    water_affinity_gain=0.35,        # extra multiplicative boost on S_w before exponent
    sand_corridor_boost=0.50,        # destination-side boost for sand near water
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

    # ------------------ EXODUS (new) ------------------
    exodus_enabled=True,
    exodus_prob=0.06,                # chance each year to trigger exodus events
    exodus_events_mean=1.0,          # Poisson mean #events when triggered
    exodus_min_pop=None,             # None → defaults to town_threshold
    exodus_group_frac=0.25,          # fraction of source pop that leaves
    exodus_min_dist=20,              # Chebyshev distance (cells)
    exodus_max_dist=None,            # optional max distance (None=unbounded)
    exodus_target_bias_water=1.0,    # 0=ignore water, 1=prefer nearer water using S_w
    exodus_target_bias_empty=0.7,    # 0=ignore emptiness, >0=prefer emptier (uses 1 - S_pop)
    exodus_culture_factor=0.5,       # seed culture counter with this * culture_persist_years
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

    # ---- histories ----
    pop_history = []
    att_history = [] if record_attractiveness else None

    def maybe_store(t, arr_pop, arr_att=None):
        if t % save_every == 0:
            pop_history.append(arr_pop.copy())
            if record_attractiveness and (arr_att is not None):
                att_history.append(arr_att.copy())

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
    maybe_store(0, N, A_att0)

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

        # ------------------ EXODUS (new) ------------------
        if exodus_enabled and rng.random() < exodus_prob:
            k_events = max(1, rng.poisson(exodus_events_mean))
            min_pop = float(town_threshold if exodus_min_pop is None else exodus_min_pop)
            # Use S_pop (from current rho) to bias toward emptier targets if requested
            # S_pop computed earlier from N (pre-flows); recompute quickly for N_next for better target choice:
            rho_next = gaussian_filter(N_next.astype(np.float32), sigma=density_sigma, mode="reflect").astype(dtype)
            S_pop_next = rho_next / (rho_next + dtype(rho_half))

            # candidate sources: sufficiently large, on land
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
                        if src_pop < min_pop:  # re-check (may have changed)
                            continue
                        group = max(1.0, exodus_group_frac * src_pop)
                        N_next[sy, sx] = max(0.0, src_pop - group)

                        # target mask: land, far enough (Chebyshev), optionally <= max
                        dist = np.maximum(np.abs(yy - sy), np.abs(xx - sx))
                        target_mask = (~water_mask) & (dist >= int(exodus_min_dist))
                        if exodus_max_dist is not None:
                            target_mask &= (dist <= int(exodus_max_dist))

                        # weights: ignore biome preference; optionally prefer near-water and emptier cells
                        if exodus_target_bias_water > 0 or exodus_target_bias_empty > 0:
                            w = np.ones((H, W), dtype=np.float64)
                            if exodus_target_bias_water > 0:
                                w *= (1e-6 + S_w) ** float(exodus_target_bias_water)
                            if exodus_target_bias_empty > 0:
                                w *= (1e-6 + (1.0 - S_pop_next)) ** float(exodus_target_bias_empty)
                        else:
                            w = np.ones((H, W), dtype=np.float64)

                        w *= target_mask.astype(np.float64)
                        wsum = w.sum()
                        if wsum <= 0:
                            # fallback: any land far enough, uniform
                            fallback = np.where(target_mask.ravel())[0]
                            if fallback.size == 0:
                                # nothing valid, return group to source
                                N_next[sy, sx] += group
                                continue
                            tidx = int(rng.choice(fallback))
                        else:
                            w = (w / wsum).ravel()
                            tidx = int(rng.choice(H*W, p=w))

                        ty, tx = divmod(tidx, W)
                        N_next[ty, tx] += group

                        # accelerated culture at the destination
                        if culture_enabled:
                            boost = int(max(0, round(exodus_culture_factor * culture_persist_years)))
                            # Seed the counter so it “arrives” earlier to anchorhood
                            culture_counter[ty, tx] = max(culture_counter[ty, tx], boost)
        # ---------------------------------------------------

        # ===== CULTURE: update & enforce floor =====
        if culture_enabled:
            size = 2 * int(culture_radius) + 1
            local_max = maximum_filter(N, size=size, mode="reflect")
            stable = local_max >= float(town_threshold)
            culture_counter = np.where(stable, culture_counter + 1, 0)
            culture_anchor = culture_anchor | (culture_counter >= int(culture_persist_years))
            floor = np.maximum(culture_min_abs, culture_min_frac * K_eff)
            N_next = np.where(culture_anchor & (~water_mask), np.maximum(N_next, floor), N_next)

        # Expire shocks
        if active_shocks:
            for sh in active_shocks: sh["years_left"] -= 1
            active_shocks = [sh for sh in active_shocks if sh["years_left"] > 0]

        N = N_next.astype(dtype, copy=False)
        maybe_store(t, N, A_att)

    # <<< END OF LOOP — returns must be AFTER the loop >>>
    if return_meta:
        meta = dict(
            biome=biome, water_mask=water_mask, D=D, land_scaled=land_scaled, h_raw=h_raw,
            culture_anchor=(culture_anchor if culture_enabled else None)
        )
        if record_attractiveness:
            return (pop_history, att_history), meta
        return pop_history, meta

    if record_attractiveness:
        return (pop_history, att_history)
    return pop_history

# =========================================================
# Visualizer (biome base + high-pop overlay + totals, population-only, attractiveness)
# =========================================================
def visualize_population_with_totals(
    pop_history,
    meta=None,
    # ---- view mode knob ----
    view="biome_overlay",            # "biome_overlay" | "population" | "attractiveness"
    # ---- overlay knobs (biome_overlay) ----
    highlight_frac=0.50,
    highlight_basis="start_max",     # "start_max" | "global_max" | "frame_max"
    overlay_gamma=0.6,
    overlay_cmap="magma",
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
    att_vmax_mode="global",          # "global" | "frame"
    # ---- animation & layout ----
    fps=12,
    title_prefix="Population — Year ",
    save_path=None,
    show=True,
    first_year=0,
    step=1,
):
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

    if view == "biome_overlay":
        if meta is None or "biome" not in meta:
            raise ValueError("meta with 'biome' is required for biome-colored base when view='biome_overlay'.")
        biome = meta["biome"].astype(np.int16)

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
        ax_top.imshow(biome, cmap=biome_cmap, norm=biome_norm, origin="upper")

        if highlight_basis not in {"start_max", "global_max", "frame_max"}:
            raise ValueError("highlight_basis must be one of: 'start_max','global_max','frame_max'.")

        def make_overlay(pop, thr, top):
            rng = max(1e-9, (top - thr))
            strength = np.clip((pop - thr) / rng, 0.0, 1.0)
            alpha = strength ** overlay_gamma
            cmap = plt.get_cmap(overlay_cmap)
            rgba = cmap(np.clip(pop / max(1e-9, top), 0, 1))
            rgba[..., 3] = alpha
            return rgba

        pop0   = pop_history[0].astype(np.float32)
        basis0 = {"start_max": start_max, "global_max": global_max, "frame_max": float(pop0.max())}[highlight_basis]
        thr0   = highlight_frac * (basis0 if basis0 > 0 else 1.0)
        top0   = basis0 if highlight_basis in ("start_max", "global_max") else float(pop0.max())
        im_top = ax_top.imshow(make_overlay(pop0, thr0, top0), origin="upper")

        def update_top(i):
            pop = pop_history[i].astype(np.float32)
            basis = {"start_max": start_max, "global_max": global_max, "frame_max": float(pop.max())}[highlight_basis]
            thr = highlight_frac * (basis if basis > 0 else 1.0)
            top = basis if highlight_basis in ("start_max", "global_max") else float(pop.max())
            im_top.set_data(make_overlay(pop, thr, top))
            return (im_top,)

        update_top(0)

    elif view == "population":
        pop0 = to_log(pop_history[0].astype(np.float32), pop_log)
        if pop_vmax is not None:
            vmax0 = pop_vmax
        else:
            vmax0 = float(max(np.max(to_log(f.astype(np.float32), pop_log)) for f in pop_history)) \
                    if pop_vmax_mode == "global" else float(pop0.max())
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
            raise ValueError("att_history is required when view='attractiveness'. Enable `record_attractiveness=True` in the simulator and pass the returned list here.")
        att0 = to_log(att_history[0].astype(np.float32), att_log)
        if att_vmax is not None:
            vmax0 = att_vmax
        else:
            vmax0 = float(max(np.max(to_log(a.astype(np.float32), att_log)) for a in att_history)) \
                    if att_vmax_mode == "global" else float(att0.max())
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

    else:
        raise ValueError("view must be 'biome_overlay' or 'population' or 'attractiveness'.")

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

    (pops, atts), meta = simulate_population_from_heightmap(
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
        # Exodus (new)
        exodus_enabled=True,
        exodus_prob=0.07,
        exodus_events_mean=1.2,
        exodus_group_frac=0.22,
        exodus_min_dist=50,
        exodus_max_dist=None,
        exodus_target_bias_water=0.0,
        exodus_target_bias_empty=0.8,
        exodus_culture_factor=0.5,
        # recording & performance
        save_every=1, dtype=np.float32, return_meta=True, record_attractiveness=True
    )
    print("Frames:", len(pops), "Grid:", pops[0].shape, "Start:", int(pops[0].sum()), "End:", int(pops[-1].sum()))

    # Try different views:
    visualize_population_with_totals(pops, meta, view="biome_overlay", highlight_frac=0.1, fps=12)
    # visualize_population_with_totals(pops, meta, view="population", pop_log=True, pop_vmax_mode="frame", fps=12)
    # visualize_population_with_totals(pops, meta, view="attractiveness", att_history=atts, att_cmap="viridis", att_log=False, att_vmax_mode="global", fps=12, first_year=0, step=1)
    
