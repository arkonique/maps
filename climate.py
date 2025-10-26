#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
climate_surface_api.py
---------------------------------
2-D surface weather toy model (CPU-fast, NumPy + Numba)
State per cell: [u, v, eta, T, q]
- Shallow-water core (u,v,eta) with terrain forcing
- Surface temperature (T) with Newtonian cooling + latent heating
- Specific humidity (q) with evap/condense + orographic enhancement
- Periodic boundaries, RK3 (Shu–Osher), flux-form upwind advection

Outputs: pure NumPy arrays you can pass to an API.
Includes a CLI main and a matplotlib visualizer/animator.
"""

import os
import math
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from numba import njit

# Optional image IO for heightmap/wetness inputs
try:
    import imageio.v3 as iio
    HAVE_IMAGEIO = True
except Exception:
    HAVE_IMAGEIO = False


# =========================================================
# Periodic stencil utilities (Numba-friendly)
# =========================================================

@njit(fastmath=True, cache=True)
def roll_x(a, k):  # k>0 => shift right by k (periodic)
    nx = a.shape[1]
    k = k % nx
    if k == 0:
        return a.copy()
    out = np.empty_like(a)
    out[:, k:] = a[:, :nx-k]
    out[:, :k] = a[:, nx-k:]
    return out

@njit(fastmath=True, cache=True)
def roll_y(a, k):  # k>0 => shift down by k (periodic)
    ny = a.shape[0]
    k = k % ny
    if k == 0:
        return a.copy()
    out = np.empty_like(a)
    out[k:, :] = a[:ny-k, :]
    out[:k, :] = a[ny-k:, :]
    return out

@njit(fastmath=True, cache=True)
def grad_central(a, dx, dy):
    axp = roll_x(a, -1); axm = roll_x(a, 1)
    ayp = roll_y(a, -1); aym = roll_y(a, 1)
    gx = (axp - axm) * (0.5/dx)
    gy = (ayp - aym) * (0.5/dy)
    return gx, gy

@njit(fastmath=True, cache=True)
def laplacian(a, dx, dy):
    axp = roll_x(a, -1); axm = roll_x(a, 1)
    ayp = roll_y(a, -1); aym = roll_y(a, 1)
    return (axp + axm - 2.0*a)/(dx*dx) + (ayp + aym - 2.0*a)/(dy*dy)

# -----------------------------
# Flux-form 1st-order upwind advection: ∇·(u A, v A)
# -----------------------------
@njit(fastmath=True, cache=True)
def flux_div_upwind(A, u, v, dx, dy):
    # x-face donor values
    A_L = A
    A_R = roll_x(A, -1)
    u_face = 0.5*(u + roll_x(u, -1))
    A_adv_x = np.where(u_face >= 0.0, A_L, A_R)
    F_x = u_face * A_adv_x
    F_xL = roll_x(F_x, 1)
    dFdx = (F_x - F_xL) / dx

    # y-face donor values
    A_B = A
    A_T = roll_y(A, -1)
    v_face = 0.5*(v + roll_y(v, -1))
    A_adv_y = np.where(v_face >= 0.0, A_B, A_T)
    F_y = v_face * A_adv_y
    F_yB = roll_y(F_y, 1)
    dFdy = (F_y - F_yB) / dy

    return dFdx + dFdy

@njit(fastmath=True, cache=True)
def div_flux(Fx, Fy, dx, dy):
    FxL = roll_x(Fx, 1)
    FyB = roll_y(Fy, 1)
    return (Fx - FxL)/dx + (Fy - FyB)/dy


# =========================================================
# Thermodynamics closures
# =========================================================

@njit(fastmath=True, cache=True)
def qsat_T(T, p_s=101000.0):
    # Saturation specific humidity at surface pressure p_s (Pa)
    T0 = 273.15
    Lv, Rv, eps = 2.5e6, 461.0, 0.622
    invT = 1.0 / np.maximum(T, 200.0)
    es = 611.0 * np.exp((Lv/Rv)*(1.0/T0 - invT))
    es = np.minimum(es, 0.99*p_s)
    return (eps*es) / (p_s - (1.0 - eps)*es)

@njit(fastmath=True, cache=True)
def make_Teq_meridional(T_sea, dTeq_dy, y_coords, h, lapse_K_per_km):
    # Simple equilibrium: meridional gradient + lapse with heightmap
    ny = h.shape[0]
    y0 = 0.5*(y_coords[0] + y_coords[ny-1])
    Teq = T_sea + dTeq_dy*(y_coords[:, None] - y0) - (lapse_K_per_km/1000.0)*h
    return Teq


# =========================================================
# One RK3 step for all fields (low-storage Shu–Osher)
# =========================================================

@njit(fastmath=True, cache=True)
def rk3_step(u, v, eta, T, q,
             hx, hy, S, alpha_qsurf,
             H, g, f0,
             r_lin, C_d, nu, kappa_h, kappa_T, kappa_q,
             cp, Lv, tau_rad, T_eq,
             tau_c, gamma_orog, C_E, p_s,
             dx, dy, dt):
    a2, a3 = 3.0/4.0, 1.0/3.0
    b1, b2, b3 = 1.0, 1.0/4.0, 2.0/3.0

    for stage in range(3):
        # Advection (flux-form upwind)
        adv_u = flux_div_upwind(u, u, v, dx, dy)
        adv_v = flux_div_upwind(v, u, v, dx, dy)
        adv_T = flux_div_upwind(T, u, v, dx, dy)
        adv_q = flux_div_upwind(q, u, v, dx, dy)

        # Pressure gradients
        etax, etay = grad_central(eta, dx, dy)

        # Diffusion / viscosity
        lap_u = laplacian(u, dx, dy)
        lap_v = laplacian(v, dx, dy)
        lap_eta = laplacian(eta, dx, dy)
        lap_T = laplacian(T, dx, dy)
        lap_q = laplacian(q, dx, dy)

        # Speeds & stresses
        speed = np.hypot(u, v)

        # Momentum RHS
        u_rhs = -adv_u + f0*v - g*(etax + hx) - r_lin*u - C_d*speed*u + nu*lap_u
        v_rhs = -adv_v - f0*u - g*(etay + hy) - r_lin*v - C_d*speed*v + nu*lap_v

        # Layer thickness/pressure surrogate
        Hu = (H + eta) * u
        Hv = (H + eta) * v
        eta_rhs = -div_flux(Hu, Hv, dx, dy) + kappa_h*lap_eta

        # Moist physics
        qsat = qsat_T(T, p_s)

        # Orographic lift proxy (ascending winds)
        w_orog = u*hx + v*hy
        w_orog = np.where(w_orog > 0.0, w_orog, 0.0)

        C_base = np.where(q > qsat, (q - qsat)/tau_c, 0.0)
        C_orog = gamma_orog * np.maximum(q, 0.0) * w_orog
        C = C_base + C_orog

        # Evaporation (bulk, capped by local qsurf cap)
        qsurf_cap = alpha_qsurf * S
        qsurf = np.minimum(qsat, qsurf_cap)
        E = C_E * speed * np.maximum(0.0, qsurf - q)

        # Tracer RHS
        T_rhs = -adv_T + kappa_T*lap_T + (T_eq - T)/tau_rad + (Lv/cp)*C
        q_rhs = -adv_q + kappa_q*lap_q + E - C

        # RK3 updates
        if stage == 0:
            u = u + b1*dt*u_rhs
            v = v + b1*dt*v_rhs
            eta = eta + b1*dt*eta_rhs
            T = T + b1*dt*T_rhs
            q = q + b1*dt*q_rhs
        elif stage == 1:
            u = a2*u + b2*(u + dt*u_rhs)
            v = a2*v + b2*(v + dt*v_rhs)
            eta = a2*eta + b2*(eta + dt*eta_rhs)
            T = a2*T + b2*(T + dt*T_rhs)
            q = a2*q + b2*(q + dt*q_rhs)
        else:
            u = a3*u + b3*(u + dt*u_rhs)
            v = a3*v + b3*(v + dt*v_rhs)
            eta = a3*eta + b3*(eta + dt*eta_rhs)
            T = a3*T + b3*(T + dt*T_rhs)
            q = a3*q + b3*(q + dt*q_rhs)

    return u, v, eta, T, q


# =========================================================
# Public API (returns NumPy arrays)
# =========================================================

def simulate_surface(
    heightmap: np.ndarray,
    wetness: np.ndarray = None,
    *,
    steps: int = 1000,
    dt: float = 10.0,
    dx: float = 2500.0,
    dy: float = 2500.0,
    save_every: int = 50,
    return_final_only: bool = False,
    # Physical params
    g: float = 9.81,
    H: float = 250.0,
    f0: float = 1.0e-4,
    r_lin: float = 2.0e-6,
    C_d: float = 1.5e-3,
    nu: float = 100.0,
    kappa_h: float = 50.0,
    kappa_T: float = 100.0,
    kappa_q: float = 100.0,
    cp: float = 1004.0,
    Lv: float = 2.5e6,
    tau_rad: float = 10.0*86400.0,  # 10 days
    tau_c: float = 2.0*3600.0,      # 2 hours
    gamma_orog: float = 5.0e-4,
    C_E: float = 1.3e-3,
    p_s: float = 101000.0,
    # Initialization / forcing
    T_sea: float = 290.0,
    lapse_K_per_km: float = 6.0,
    meridional_dT_per_m: float = -30.0/1.0e6,
    # Evap cap scaling (turn wetness into qsurf cap)
    alpha_soil_scale: float = 0.012,
    dtype=np.float32
):
    """
    Returns:
      - if return_final_only: (ny, nx, 5) final frame [u,v,eta,T,q]
      - else: (nt, ny, nx, 5) stacked snapshots every save_every steps
    """
    h = heightmap.astype(dtype, copy=False)
    ny, nx = h.shape

    # Wetness S in [0,1]: if not provided, infer from elevation (water near z<=5m)
    if wetness is None:
        S = np.exp(-np.maximum(h - 5.0, 0.0)/300.0).astype(dtype)
    else:
        S = np.clip(wetness.astype(dtype, copy=False), 0.0, 1.0)

    # Terrain slopes
    hx, hy = grad_central(h, dx, dy)

    # Equilibrium temperature (meridional gradient + lapse)
    y_coords = np.arange(ny, dtype=np.float64) * dy
    T_eq = make_Teq_meridional(T_sea, meridional_dT_per_m, y_coords, h, lapse_K_per_km).astype(dtype)

    # Initial state
    T = T_eq.copy()
    q = (0.5 * qsat_T(T, p_s)).astype(dtype)   # ~50% RH
    u = np.zeros_like(h, dtype=dtype)
    v = np.zeros_like(h, dtype=dtype)
    eta = np.zeros_like(h, dtype=dtype)

    # Evap cap map (kg/kg): alpha_soil_scale * qsat(T_sea)
    qsat_sea = qsat_T(np.full_like(h, T_sea, dtype=dtype), p_s).astype(dtype)
    alpha_qsurf = alpha_soil_scale * qsat_sea

    # CFL guidance (gravity-wave speed)
    c = math.sqrt(g*H)
    dt_cfl = 0.45 * min(dx, dy) / c
    if dt > dt_cfl:
        dt = dt_cfl  # gently clip for safety

    if return_final_only:
        for _ in range(steps):
            u, v, eta, T, q = rk3_step(
                u, v, eta, T, q,
                hx, hy, S, alpha_qsurf,
                H, g, f0,
                r_lin, C_d, nu, kappa_h, kappa_T, kappa_q,
                cp, Lv, tau_rad, T_eq,
                tau_c, gamma_orog, C_E, p_s,
                dx, dy, dt
            )
        return np.stack((u, v, eta, T, q), axis=-1)

    nt = (steps // save_every) + 1
    out = np.empty((nt, ny, nx, 5), dtype=dtype)
    t_index = 0
    out[t_index, ...] = np.stack((u, v, eta, T, q), axis=-1)
    t_index += 1

    for n in range(1, steps+1):
        u, v, eta, T, q = rk3_step(
            u, v, eta, T, q,
            hx, hy, S, alpha_qsurf,
            H, g, f0,
            r_lin, C_d, nu, kappa_h, kappa_T, kappa_q,
            cp, Lv, tau_rad, T_eq,
            tau_c, gamma_orog, C_E, p_s,
            dx, dy, dt
        )
        if (n % save_every) == 0:
            out[t_index, ...] = np.stack((u, v, eta, T, q), axis=-1)
            t_index += 1

    return out


# =========================================================
# Simple visualizer/animator
# =========================================================

def visualize_states(states, heightmap=None, view="combo",
                     quiver_step=8, fps=15, save_path=None, show=True,
                     cmap_T="coolwarm", cmap_q="Blues", cmap_eta="PuOr",
                     title="Surface model", vminmax_T=None, vminmax_q=None, vminmax_eta=None):
    """
    states: np.ndarray (nt, ny, nx, 5) with [u,v,eta,T,q]
    view: "combo" | "T" | "q" | "eta" | "winds"
    """
    assert states.ndim == 4 and states.shape[-1] == 5
    nt, ny, nx, _ = states.shape
    u0, v0, eta0, T0, q0 = [states[0, ..., i] for i in range(5)]

    # Ranges
    if vminmax_T is None:
        vminmax_T = (float(np.nanmin(states[..., 3])), float(np.nanmax(states[..., 3])))
    if vminmax_q is None:
        vminmax_q = (0.0, float(np.nanmax(states[..., 4])))
    if vminmax_eta is None:
        vminmax_eta = (-float(np.nanmax(np.abs(states[..., 2]))), float(np.nanmax(np.abs(states[..., 2]))))

    # Figure
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    ax.set_xticks([]); ax.set_yticks([])
    ttl = ax.set_title(title)

    # Background: heightmap shaded (optional)
    if heightmap is not None and view in ("combo",):
        hm = (heightmap - np.min(heightmap)) / (np.ptp(heightmap) + 1e-9)
        ax.imshow(hm, cmap="gray", alpha=0.35, origin="upper")

    # Layers
    im_T = im_q = im_eta = None
    Q = None

    if view in ("combo", "T"):
        im_T = ax.imshow(T0, cmap=cmap_T, vmin=vminmax_T[0], vmax=vminmax_T[1], alpha=0.6, origin="upper")
        plt.colorbar(im_T, ax=ax, fraction=0.046, pad=0.02, label="Temperature (K)")
    if view in ("combo", "q"):
        im_q = ax.imshow(q0, cmap=cmap_q, vmin=vminmax_q[0], vmax=vminmax_q[1], alpha=0.45 if view=="combo" else 1.0, origin="upper")
        plt.colorbar(im_q, ax=ax, fraction=0.046, pad=0.02, label="Specific humidity (kg/kg)")
    if view in ("eta",):
        im_eta = ax.imshow(eta0, cmap=cmap_eta, vmin=vminmax_eta[0], vmax=vminmax_eta[1], origin="upper")
        plt.colorbar(im_eta, ax=ax, fraction=0.046, pad=0.02, label="η (m)")

    if view in ("combo", "winds",):
        step = max(1, quiver_step)
        Y, X = np.mgrid[0:ny:step, 0:nx:step]
        Q = ax.quiver(X, Y, u0[::step, ::step], v0[::step, ::step], color="k", scale=200, width=0.002, alpha=0.8)

    def update(k):
        u, v, eta, T, q = [states[k, ..., i] for i in range(5)]
        if im_T is not None: im_T.set_data(T)
        if im_q is not None: im_q.set_data(q)
        if im_eta is not None: im_eta.set_data(eta)
        if Q is not None:
            Q.set_UVC(u[::quiver_step, ::quiver_step], v[::quiver_step, ::quiver_step])
        ttl.set_text(f"{title} — frame {k+1}/{nt}")
        return (im_T, im_q, im_eta, Q, ttl)

    anim = FuncAnimation(fig, update, frames=nt, interval=1000/max(1, fps), blit=False)

    if save_path:
        ext = os.path.splitext(save_path)[1].lower()
        if ext in (".mp4", ".gif"):
            anim.save(save_path, dpi=140)
        else:
            print(f"[warn] save_path '{save_path}' has unknown extension; skipping save.")

    if show:
        plt.show()
    plt.close(fig)
    return anim


# =========================================================
# IO helpers
# =========================================================

def load_array_or_image(path, scale_to=(0.0, 3000.0), as_float=True):
    """Load .npy or image; returns float array (meters) normalized to chosen range if image."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        arr = np.load(path)
        return arr.astype(np.float32) if as_float else arr
    if not HAVE_IMAGEIO:
        raise RuntimeError("Install imageio (pip install imageio) to read images, or provide a .npy.")
    img = iio.imread(path)
    if img.ndim == 3:
        img = img[..., :3].dot(np.array([0.299, 0.587, 0.114], dtype=np.float32))
    img = img.astype(np.float32)
    # Normalize 0..1 then scale to meters
    vmin, vmax = np.min(img), np.max(img)
    if vmax > vmin:
        img = (img - vmin) / (vmax - vmin)
    z0, z1 = scale_to
    arr = z0 + img * (z1 - z0)
    return arr


# =========================================================
# Main (CLI)
# =========================================================

def parse_kv(s):
    k, v = s.split("=", 1)
    try:
        v_parsed = json.loads(v)
    except Exception:
        try:
            v_parsed = float(v) if ("." in v or "e" in v.lower()) else int(v)
        except Exception:
            v_parsed = v
    return k, v_parsed

def main():
    ap = argparse.ArgumentParser(description="2-D surface model (NumPy+Numba) — returns NumPy arrays; optional viz.")
    ap.add_argument("heightmap", help=".npy (float meters) or image file (will be scaled to meters)")
    ap.add_argument("--wetness", help=".npy/image [0..1], 1=water/wet, 0=desert", default=None)
    ap.add_argument("--scale", help="If image heightmap, scale range in meters, e.g. --scale 0 4000", nargs=2, type=float, default=[0.0, 3000.0])
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--dt", type=float, default=10.0)
    ap.add_argument("--dx", type=float, default=2500.0)
    ap.add_argument("--dy", type=float, default=2500.0)
    ap.add_argument("--save-every", type=int, default=50)
    ap.add_argument("--final-only", action="store_true", help="Return only the final frame")
    ap.add_argument("--npz", help="Optional path to save states as NPZ (purely optional)", default=None)
    ap.add_argument("--viz", action="store_true", help="Show an animation preview")
    ap.add_argument("--viz-view", default="combo", choices=["combo","T","q","eta","winds"])
    ap.add_argument("--viz-fps", type=int, default=15)
    ap.add_argument("--viz-save", help="Optional .mp4 or .gif to save animation", default=None)
    ap.add_argument("--set", action="append", default=[], help="Override any simulate_surface kwarg, e.g. --set H=300 --set tau_c=5400")

    args = ap.parse_args()

    # Load inputs
    h = load_array_or_image(args.heightmap, scale_to=tuple(args.scale))
    S = None
    if args.wetness:
        S = load_array_or_image(args.wetness, scale_to=(0.0, 1.0))
        S = np.clip(S, 0.0, 1.0).astype(np.float32)

    # Build overrides dict for simulate_surface
    overrides = dict(steps=args.steps, dt=args.dt, dx=args.dx, dy=args.dy,
                     save_every=args.save_every, return_final_only=args.final_only)

    # Parse --set overrides
    for s in args.set:
        if s.strip().startswith("{"):
            overrides.update(json.loads(s))
        else:
            k, v = parse_kv(s)
            overrides[k] = v

    # Run simulation
    states = simulate_surface(h, S, **overrides)

    # Optionally save NPZ (purely optional; you said you want NumPy arrays available)
    if args.npz:
        if states.ndim == 3:
            np.savez_compressed(args.npz, final=states)
        else:
            np.savez_compressed(args.npz, states=states)
        print(f"[save] {args.npz}")

    # Optional viz
    if args.viz:
        title = f"Surface model — {os.path.basename(args.heightmap)}"
        if states.ndim == 3:
            # single frame — expand to 2 frames for a tiny animation
            states = np.stack([states, states], axis=0)
        visualize_states(states, heightmap=h, view=args.viz_view,
                         fps=args.viz_fps, save_path=args.viz_save,
                         title=title, quiver_step=max(4, int(min(h.shape)/128)))

    # If run as script, still expose arrays in-memory for caller processes (if imported, they can call simulate_surface)

if __name__ == "__main__":
    main()
