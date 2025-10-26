# =========================================================
#  Procedural Voronoi World Map
#  • NumPy 2.0+ safe • multi-mode heightmap • lakes + mountains
#  • Terrain fixes: vignette (edge falloff), continent bias, landmass recenter
#  • Rivers: diverse sources & mouths, count knob, meander/jaggedness knobs
#  • Rivers trimmed exactly at coastline, optional carving
#  • BITMAP renderer (per-pixel + fast blur) with COASTLINE + RIVER overlays
#  • POLYGON renderer (Voronoi + optional domain warp) preserved
#  • No matplotlib, no skimage
# =========================================================
import numpy as np
from scipy.spatial import Voronoi
import plotly.graph_objects as go
from dataclasses import dataclass
from typing import List, Tuple, Optional

# =========================
# Data structure
# =========================
@dataclass
class VoronoiMesh:
    points: np.ndarray
    regions: List[np.ndarray]
    width_scale: float
    length_scale: float
    bbox: Tuple[float, float, float, float]


# =========================
# Voronoi construction
# =========================
def _finite_polygons(vor: Voronoi, radius=None):
    """Convert infinite Voronoi regions to finite polygons (NumPy 2.0 safe)."""
    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input.")
    new_vertices = vor.vertices.tolist()
    center = vor.points.mean(axis=0)
    if radius is None:
        radius = np.ptp(vor.points, axis=0).max() * 2.0

    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(int(p1), []).append((int(p2), int(v1), int(v2)))
        all_ridges.setdefault(int(p2), []).append((int(p1), int(v1), int(v2)))

    new_regions = []
    for p1, reg_idx in enumerate(vor.point_region):
        region = vor.regions[reg_idx] if reg_idx != -1 else None
        if region and -1 not in region:
            new_regions.append(region)
            continue

        new_reg = [v for v in (region or []) if v >= 0]
        for p2, v1, v2 in all_ridges.get(p1, []):
            if v1 >= 0 and v2 >= 0:
                continue
            t = vor.points[p2] - vor.points[p1]
            nrm = np.linalg.norm(t)
            if nrm == 0:
                continue
            t /= nrm
            n = np.array([-t[1], t[0]])
            midpoint = 0.5 * (vor.points[p1] + vor.points[p2])
            direction = np.sign((midpoint - center).dot(n)) * n
            if v1 >= 0:
                v_finite = vor.vertices[v1]
            elif v2 >= 0:
                v_finite = vor.vertices[v2]
            else:
                v_finite = midpoint
            far_pt = v_finite + direction * radius
            new_vertices.append(far_pt.tolist())
            new_reg.append(len(new_vertices) - 1)

        vs = np.asarray([new_vertices[v] for v in new_reg], dtype=float)
        if vs.size == 0:
            continue
        c = vs.mean(axis=0)
        ang = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
        new_regions.append([v for _, v in sorted(zip(ang, new_reg))])

    return new_regions, np.asarray(new_vertices)


def build_voronoi(num_points=6000, width=1600.0, length=900.0, seed=42) -> VoronoiMesh:
    rng = np.random.default_rng(seed)
    pts = np.column_stack([
        rng.random(num_points) * width,
        rng.random(num_points) * length
    ])
    vor = Voronoi(pts)
    regions, verts = _finite_polygons(vor)
    polys = []
    for r in regions:
        if not r:
            continue
        poly = verts[np.asarray(r)]
        if (
            np.all((0 <= poly[:, 0]) & (poly[:, 0] <= width)) and
            np.all((0 <= poly[:, 1]) & (poly[:, 1] <= length)) and
            len(poly) >= 3
        ):
            polys.append(poly.astype(float))
    return VoronoiMesh(pts, polys, width, length, (0, 0, width, length))


# =========================
# Noise primitives
# =========================
def _rng(seed: Optional[int]):
    return np.random.default_rng() if seed is None else np.random.default_rng(int(seed))


def _fft_noise(w: int, h: int, beta: float = 2.0, seed: Optional[int] = None):
    """
    Power-law spectral noise (tileable due to FFT). We normalize but DO NOT
    leave a strong DC term, so we don't bias to flat slabs.
    """
    rng = _rng(seed)
    kx = np.fft.fftfreq(w)
    ky = np.fft.fftfreq(h)
    kx2, ky2 = np.meshgrid(kx, ky)
    k2 = kx2**2 + ky2**2
    k2[0, 0] = 1.0
    amp = 1.0 / (k2 ** (beta / 2.0))
    # slightly de-emphasize the very lowest frequencies to avoid giant blobs
    amp *= 1.0 / (1.0 + 8.0 * k2)

    phase = rng.uniform(0, 2 * np.pi, size=(h, w))
    field = amp * np.exp(1j * phase)
    arr = np.fft.ifft2(field).real
    arr -= arr.min()
    arr /= arr.max() + 1e-12
    return arr


def _bilinear_resize(img: np.ndarray, out_w: int, out_h: int):
    in_h, in_w = img.shape
    y = np.linspace(0, in_h - 1, out_h)
    x = np.linspace(0, in_w - 1, out_w)
    x0 = np.floor(x).astype(int); x1 = np.clip(x0 + 1, 0, in_w - 1)
    y0 = np.floor(y).astype(int); y1 = np.clip(y0 + 1, 0, in_h - 1)
    sx = (x - x0)[None, :]; sy = (y - y0)[:, None]
    return (
        img[y0[:, None], x0[None, :]] * (1 - sx) * (1 - sy) +
        img[y0[:, None], x1[None, :]] * sx * (1 - sy) +
        img[y1[:, None], x0[None, :]] * (1 - sx) * sy +
        img[y1[:, None], x1[None, :]] * sx * sy
    )


def _resampled_noise(w: int, h: int, scale: float, beta: float, seed: Optional[int]):
    sw = max(8, int(round(w / max(1e-6, scale))))
    sh = max(8, int(round(h / max(1e-6, scale))))
    small = _fft_noise(sw, sh, beta=beta, seed=seed)
    big = _bilinear_resize(small, w, h)
    big -= big.min()
    big /= big.max() + 1e-12
    return big


# =========================
# Box blurs (scalar + RGB), shape-safe
# =========================
def _box_blur(img: np.ndarray, k: int = 3) -> np.ndarray:
    k = int(max(0, k))
    if k == 0:
        return img.astype(np.float64, copy=True)
    src = img.astype(np.float64, copy=False)
    H, W = src.shape
    win = 2 * k + 1
    hpad = np.pad(src, ((0, 0), (k, k)), mode="edge")
    cs = np.cumsum(hpad, axis=1)
    cs = np.pad(cs, ((0, 0), (1, 0)), mode="constant")
    out_h = (cs[:, win:] - cs[:, :-win]) / win
    vpad = np.pad(out_h, ((k, k), (0, 0)), mode="edge")
    cs2 = np.cumsum(vpad, axis=0)
    cs2 = np.pad(cs2, ((1, 0), (0, 0)), mode="constant")
    out_v = (cs2[win:, :] - cs2[:-win, :]) / win
    return out_v


def _box_blur_rgb(img: np.ndarray, k: int = 2) -> np.ndarray:
    k = int(max(0, k))
    if k == 0:
        return img.astype(np.float64, copy=True)
    if img.dtype.kind in ("u", "i"):
        src = img.astype(np.float64) / 255.0
    else:
        src = img.astype(np.float64)
    H, W, C = src.shape
    win = 2 * k + 1
    hpad = np.pad(src, ((0, 0), (k, k), (0, 0)), mode="edge")
    cs = np.cumsum(hpad, axis=1)
    cs = np.pad(cs, ((0, 0), (1, 0), (0, 0)), mode="constant")
    out_h = (cs[:, win:, :] - cs[:, :-win, :]) / win
    vpad = np.pad(out_h, ((k, k), (0, 0), (0, 0)), mode="edge")
    cs2 = np.cumsum(vpad, axis=0)
    cs2 = np.pad(cs2, ((1, 0), (0, 0), (0, 0)), mode="constant")
    out_v = (cs2[win:, :, :] - cs2[:-win, :, :]) / win
    return out_v


# =========================
# Domain warp for Voronoi polygons
# =========================
def _bilinear_sample_field(field: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    H, W = field.shape
    x = np.clip(xs, 0, W - 1); y = np.clip(ys, 0, H - 1)
    x0 = np.floor(x).astype(int); y0 = np.floor(ys).astype(int)
    x1 = np.clip(x0 + 1, 0, W - 1); y1 = np.clip(y0 + 1, 0, H - 1)
    sx = x - x0; sy = ys - y0
    f00 = field[y0, x0]; f10 = field[y0, x1]; f01 = field[y1, x0]; f11 = field[y1, x1]
    return (f00 * (1 - sx) * (1 - sy) + f10 * sx * (1 - sy) + f01 * (1 - sx) * sy + f11 * sx * sy)


def warp_mesh(mesh: VoronoiMesh, strength: float = 8.0, scale: float = 3.0, seed: Optional[int] = None) -> VoronoiMesh:
    if strength <= 0:
        return mesh
    grid_W = 256
    grid_H = int(max(64, round(256 * mesh.length_scale / max(1e-6, mesh.width_scale))))
    u = _resampled_noise(grid_W, grid_H, scale=scale, beta=2.4, seed=seed)
    v = _resampled_noise(grid_W, grid_H, scale=scale, beta=2.4, seed=(None if seed is None else int(seed) + 1))
    u = (u * 2 - 1)
    v = (v * 2 - 1)
    all_xy = np.concatenate([p for p in mesh.regions], axis=0)
    xs = all_xy[:, 0] / mesh.width_scale  * (grid_W - 1)
    ys = all_xy[:, 1] / mesh.length_scale * (grid_H - 1)
    du = _bilinear_sample_field(u, xs, ys) * float(strength)
    dv = _bilinear_sample_field(v, xs, ys) * float(strength)
    warped_xy = np.column_stack([
        np.clip(all_xy[:, 0] + du, 0, mesh.width_scale),
        np.clip(all_xy[:, 1] + dv, 0, mesh.length_scale)
    ])
    out_polys, k = [], 0
    for poly in mesh.regions:
        n = len(poly)
        out_polys.append(warped_xy[k:k+n])
        k += n
    return VoronoiMesh(mesh.points, out_polys, mesh.width_scale, mesh.length_scale, mesh.bbox)


# =========================
# Terrain helpers to avoid edge blobs & recenter land
# =========================
def _vignette_mask(w: int, h: int, margin_frac: float = 0.08, power: float = 2.2):
    """
    Returns [h,w] mask in [0,1] that fades toward edges. Larger margin/power => stronger edge falloff.
    """
    y = np.linspace(0, 1, h)[:, None]
    x = np.linspace(0, 1, w)[None, :]
    dx = np.minimum(x, 1 - x)
    dy = np.minimum(y, 1 - y)
    d = np.minimum(dx, dy)
    m = np.clip((d / margin_frac), 0, 1) ** power
    return m


def _continent_bias(w: int, h: int, squash: float = 0.85, falloff: float = 1.6):
    """
    Elliptical radial mask peaking near center to discourage twin edge blobs.
    squash<1 squashes vertically a bit.
    """
    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy = (w - 1) * 0.5, (h - 1) * 0.5
    nx = (xx - cx) / (0.5 * w)
    ny = (yy - cy) / (0.5 * h * max(1e-6, squash))
    r = np.sqrt(nx * nx + ny * ny)
    mask = np.clip(1.0 - r, 0.0, 1.0) ** falloff
    return mask


def _recentre_land(arr: np.ndarray, sea: float):
    """
    Recentre landmass by rolling the array so the land centroid is at image center.
    """
    H, W = arr.shape
    land = arr > sea
    if not np.any(land):
        return arr
    ys, xs = np.nonzero(land)
    cx = int(np.mean(xs))
    cy = int(np.mean(ys))
    shift_x = W // 2 - cx
    shift_y = H // 2 - cy
    return np.roll(np.roll(arr, shift_y, axis=0), shift_x, axis=1)


# =========================
# Heightmap with modes + lakes + mountains
# =========================
def generate_heightmap(
    mode: str = "world",
    width_px: int = 1024,
    length_px: int = 512,
    water_pct: float = 0.65,
    seed: Optional[int] = None,
    lakes_pct: float = 0.02,
    lakes_scale: float = 0.8,
    lakes_smooth: int = 3,
    mountain_strength: float = 0.45,
    mountain_gamma: float = 1.6,
    mountain_mask_scale: float = 3.0,
    max_land_elev: float = 1.00,
    # NEW: terrain tamers
    vignette_strength: float = 0.35,   # 0 disables edge falloff
    vignette_margin: float = 0.08,
    continent_bias_strength: float = 0.45,  # 0 disables center bias
    recentre_landmass: bool = True
):
    w, h = int(width_px), int(length_px)
    rng = _rng(seed)

    # Base spectrum shapes
    if mode == "world":
        base = _fft_noise(w, h, beta=2.1, seed=seed)
    elif mode == "continents":
        base = _fft_noise(w, h, beta=2.4, seed=seed)
    elif mode == "archipelago":
        base = _fft_noise(w, h, beta=1.8, seed=seed)
    elif mode == "islands":
        base = _fft_noise(w, h, beta=1.6, seed=seed)
        # radial falloff for island feel
        y, x = np.ogrid[0:h, 0:w]
        cy, cx = h / 2, w / 2
        r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        falloff = np.clip(1 - (r / (0.8 * min(cx, cy))) ** 2, 0, 1)
        base *= falloff
    else:
        raise ValueError(f"Unknown mode '{mode}'")

    # Apply terrain tamers (multiplicative), then renormalize
    if continent_bias_strength > 1e-6:
        base *= (1.0 - continent_bias_strength) + continent_bias_strength * _continent_bias(w, h)

    if vignette_strength > 1e-6:
        base *= (1.0 - vignette_strength) + vignette_strength * _vignette_mask(w, h, margin_frac=vignette_margin)

    base -= base.min()
    base /= base.max() + 1e-12

    sea = float(np.quantile(base, water_pct))

    # Optionally recenter the landmass for world/continents to keep it in-frame
    if recentre_landmass and mode in ("world", "continents"):
        base = _recentre_land(base, sea)
        base -= base.min(); base /= base.max() + 1e-12
        sea = float(np.quantile(base, water_pct))

    # Land normalization pipeline
    land_mask = base > sea
    land = np.where(land_mask, (base - sea) / max(1e-6, (1 - sea)), 0.0)

    # Lakes
    if lakes_pct > 1e-6 and np.any(land_mask):
        lakes_noise = _resampled_noise(w, h, scale=max(1e-6, lakes_scale), beta=2.0,
                                       seed=rng.integers(1 << 31))
        q = np.quantile(lakes_noise[land_mask], lakes_pct)
        lakes_raw = (lakes_noise <= q).astype(float)
        lakes_s = _box_blur(lakes_raw, k=max(0, int(lakes_smooth)))
        lakes_mask = (lakes_s > 0.5) & land_mask
        base = np.where(lakes_mask, np.minimum(base, sea * 0.98), base)
        land_mask = base > sea
        land = np.where(land_mask, (base - sea) / max(1e-6, (1 - sea)), 0.0)

    # Mountains
    if mountain_strength > 1e-6 and np.any(land_mask):
        ridged_src = _resampled_noise(w, h, scale=0.8, beta=1.5,
                                      seed=rng.integers(1 << 31))
        ridged = 1.0 - np.abs(ridged_src - 0.5) * 2.0
        ridged = np.clip(ridged, 0.0, 1.0) ** float(mountain_gamma)
        m_mask = _resampled_noise(w, h, scale=max(1e-6, mountain_mask_scale), beta=2.6,
                                  seed=rng.integers(1 << 31))
        m_mask = m_mask ** 1.2
        m_field = ridged * m_mask
        m_field -= m_field.min(); m_field /= (m_field.max() + 1e-12)
        boosted_land = np.clip(land + mountain_strength * m_field * (1.0 - land), 0.0, 1.0)
        boosted_land = np.minimum(boosted_land, float(max_land_elev))
        base = np.where(land_mask, sea + boosted_land * (1.0 - sea), base)

    base -= base.min(); base /= base.max() + 1e-12
    sea = float(np.quantile(base, water_pct))
    return base, sea


# =========================
# Terrain-like palette (no matplotlib)
# =========================
_LAND_STOPS = [
    (0.00, ( 70, 110,  40)),
    (0.30, (110, 150,  60)),
    (0.55, (160, 130,  90)),
    (0.80, (190, 175, 160)),
    (1.00, (255, 255, 255)),
]

def _interp_color(stops, t):
    t = float(np.clip(t, 0.0, 1.0))
    for (p0, c0), (p1, c1) in zip(stops[:-1], stops[1:]):
        if t <= p1:
            w = 0.0 if p1 == p0 else (t - p0) / (p1 - p0)
            c0 = np.asarray(c0, dtype=float); c1 = np.asarray(c1, dtype=float)
            return (c0 * (1 - w) + c1 * w) / 255.0
    return np.asarray(stops[-1][1], dtype=float) / 255.0


def _build_land_palette(stops, size: int = 2048) -> np.ndarray:
    size = int(max(2, size))
    t = np.linspace(0.0, 1.0, size, endpoint=True)
    ps = np.array([p for p, _ in stops], dtype=float)
    cs = np.array([c for _, c in stops], dtype=float) / 255.0
    idx = np.searchsorted(ps, t, side="right") - 1
    idx = np.clip(idx, 0, len(ps) - 2)
    p0 = ps[idx]; p1 = ps[idx + 1]
    c0 = cs[idx]; c1 = cs[idx + 1]
    denom = np.where(p1 == p0, 1.0, (p1 - p0))
    w = np.clip((t - p0) / denom, 0.0, 1.0)
    pal = c0 * (1.0 - w)[:, None] + c1 * w[:, None]
    pal[t >= ps[-1]] = cs[-1]
    return pal


def _colorize_terrain(vals: np.ndarray,
                      sea: float,
                      ocean_deep=(20, 60, 150),
                      ocean_shallow=(150, 200, 230)) -> np.ndarray:
    vals = np.asarray(vals, dtype=float)
    colors = np.zeros((vals.size, 3), dtype=float)

    water = vals <= sea
    if np.any(water):
        t = np.zeros_like(vals)
        if sea > 1e-12:
            t[water] = np.clip(vals[water] / sea, 0.0, 1.0)
        od = np.asarray(ocean_deep, dtype=float) / 255.0
        os = np.asarray(ocean_shallow, dtype=float) / 255.0
        colors[water] = od + (os - od) * t[water, None]

    land = ~water
    if np.any(land):
        denom = max(1.0 - sea, 1e-12)
        t = np.clip((vals[land] - sea) / denom, 0.0, 1.0)
        out = np.empty((t.shape[0], 3), dtype=float)
        for i, tv in enumerate(t):
            out[i] = _interp_color(_LAND_STOPS, float(tv))
        colors[land] = out

    return colors


def colorize_bitmap(heightmap: np.ndarray,
                    sea: float,
                    ocean_deep=(20, 60, 150),
                    ocean_shallow=(150, 200, 230),
                    land_palette_size: int = 2048) -> np.ndarray:
    H, W = heightmap.shape
    img = np.zeros((H, W, 3), dtype=float)
    od = np.asarray(ocean_deep, dtype=float) / 255.0
    os = np.asarray(ocean_shallow, dtype=float) / 255.0
    water = heightmap <= sea
    if np.any(water):
        t = np.zeros_like(heightmap, dtype=float)
        if sea > 1e-12:
            t[water] = np.clip(heightmap[water] / sea, 0.0, 1.0)
        img[water] = od + (os - od) * t[water, None]
    land = ~water
    if np.any(land):
        denom = max(1.0 - sea, 1e-12)
        t = np.clip((heightmap[land] - sea) / denom, 0.0, 1.0)
        pal = _build_land_palette(_LAND_STOPS, size=land_palette_size)
        q = np.minimum((t * (land_palette_size - 1)).astype(int), land_palette_size - 1)
        img[land] = pal[q]
    return img  # float [0,1]


# =========================
# Marching Squares: coastlines at sea level
# =========================
def _interp_iso(x1, y1, v1, x2, y2, v2, iso):
    dv = v2 - v1
    t = 0.5 if abs(dv) < 1e-12 else (iso - v1) / dv
    t = float(np.clip(t, 0.0, 1.0))
    return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))


def marching_squares_segments(field: np.ndarray, iso: float):
    H, W = field.shape
    xs, ys = [], []
    for i in range(H - 1):
        v0 = field[i, :]
        v1 = field[i+1, :]
        for j in range(W - 1):
            v00 = v0[j];     v10 = v0[j+1]
            v01 = v1[j];     v11 = v1[j+1]
            c0 = 1 if v00 > iso else 0
            c1 = 1 if v10 > iso else 0
            c2 = 1 if v11 > iso else 0
            c3 = 1 if v01 > iso else 0
            case = (c0 << 0) | (c1 << 1) | (c2 << 2) | (c3 << 3)
            if case == 0 or case == 15:
                continue

            pts = []
            if case in (1, 14, 2, 13, 4, 11, 8, 7):
                if (case in (1, 14)):
                    a = _interp_iso(j, i, v00, j+1, i, v10, iso)
                    d = _interp_iso(j, i, v00, j, i+1, v01, iso)
                    pts = [a, d]
                elif (case in (2, 13)):
                    a = _interp_iso(j, i, v00, j+1, i, v10, iso)
                    b = _interp_iso(j+1, i, v10, j+1, i+1, v11, iso)
                    pts = [a, b]
                elif (case in (4, 11)):
                    b = _interp_iso(j+1, i, v10, j+1, i+1, v11, iso)
                    c = _interp_iso(j, i+1, v01, j+1, i+1, v11, iso)
                    pts = [b, c]
                elif (case in (8, 7)):
                    c = _interp_iso(j, i+1, v01, j+1, i+1, v11, iso)
                    d = _interp_iso(j, i, v00, j, i+1, v01, iso)
                    pts = [c, d]
            elif case in (3, 12, 6, 9):
                if case == 3:
                    b = _interp_iso(j+1, i, v10, j+1, i+1, v11, iso)
                    d = _interp_iso(j, i, v00, j, i+1, v01, iso)
                    pts = [b, d]
                elif case == 12:
                    a = _interp_iso(j, i, v00, j+1, i, v10, iso)
                    c = _interp_iso(j, i+1, v01, j+1, i+1, v11, iso)
                    pts = [a, c]
                elif case == 6:
                    a = _interp_iso(j, i, v00, j+1, i, v10, iso)
                    c = _interp_iso(j, i+1, v01, j+1, i+1, v11, iso)
                    pts = [a, c]
                elif case == 9:
                    b = _interp_iso(j+1, i, v10, j+1, i+1, v11, iso)
                    d = _interp_iso(j, i, v00, j, i+1, v01, iso)
                    pts = [b, d]
            else:
                center = (v00 + v10 + v01 + v11) * 0.25
                if case == 5:
                    a = _interp_iso(j, i, v00, j+1, i, v10, iso)
                    c = _interp_iso(j, i+1, v01, j+1, i+1, v11, iso)
                    b = _interp_iso(j+1, i, v10, j+1, i+1, v11, iso)
                    d = _interp_iso(j, i, v00, j, i+1, v01, iso)
                    if center > iso:
                        pts = [a, d]
                        xs.extend([None]); ys.extend([None])
                        xs.extend([b[0], c[0]]); ys.extend([b[1], c[1]])
                        continue
                    else:
                        pts = [a, b]
                        xs.extend([None]); ys.extend([None])
                        xs.extend([d[0], c[0]]); ys.extend([d[1], c[1]])
                        continue
                elif case == 10:
                    a = _interp_iso(j, i, v00, j+1, i, v10, iso)
                    c = _interp_iso(j, i+1, v01, j+1, i+1, v11, iso)
                    b = _interp_iso(j+1, i, v10, j+1, i+1, v11, iso)
                    d = _interp_iso(j, i, v00, j, i+1, v01, iso)
                    if center > iso:
                        pts = [a, b]
                        xs.extend([None]); ys.extend([None])
                        xs.extend([d[0], c[0]]); ys.extend([d[1], c[1]])
                        continue
                    else:
                        pts = [a, d]
                        xs.extend([None]); ys.extend([None])
                        xs.extend([b[0], c[0]]); ys.extend([b[1], c[1]])
                        continue

            if pts:
                (x1, y1), (x2, y2) = pts
                xs.extend([x1, x2, None])
                ys.extend([y1, y2, None])

    return np.array(xs, dtype=float), np.array(ys, dtype=float)


# =========================
# RIVERS — utilities & core logic (Delaunay from Voronoi sites)
# =========================
def delaunay_adjacency_from_points(points: np.ndarray) -> dict[int, set[int]]:
    vor = Voronoi(points)
    adj = {i: set() for i in range(points.shape[0])}
    for i, j in vor.ridge_points:
        i, j = int(i), int(j)
        adj[i].add(j); adj[j].add(i)
    return adj


def _hash_edge(a: int, b: int) -> int:
    """Deterministic 64-bit mix for undirected edge (no NumPy overflows)."""
    if a > b:
        a, b = b, a
    k = ((a & 0xFFFFFFFF) << 32) ^ (b & 0xFFFFFFFF)
    k ^= (k >> 30)
    k = (k * 0xbf58476d1ce4e5b9) & 0xFFFFFFFFFFFFFFFF
    k ^= (k >> 27)
    k = (k * 0x94d049bb133111eb) & 0xFFFFFFFFFFFFFFFF
    k ^= (k >> 31)
    return int(k & 0x7FFFFFFF)


def find_coast_pixels(heightmap: np.ndarray, sea: float):
    water = heightmap <= sea
    land = ~water
    coast = np.zeros_like(water, dtype=bool)
    coast[:-1, :] |= water[:-1, :] & land[1:, :]
    coast[1:,  :] |= water[1:,  :] & land[:-1, :]
    coast[:, :-1] |= water[:, :-1] & land[:, 1:]
    coast[:, 1:]  |= water[:, 1:]  & land[:, :-1]
    return coast


def sample_scalar_bilinear(field: np.ndarray,
                           xs_world: np.ndarray,
                           ys_world: np.ndarray,
                           width_scale: float,
                           length_scale: float) -> np.ndarray:
    H, W = field.shape
    xs = np.clip(xs_world / width_scale  * (W - 1), 0, W - 1)
    ys = np.clip(ys_world / length_scale * (H - 1), 0, H - 1)
    x0 = np.floor(xs).astype(int); y0 = np.floor(ys).astype(int)
    x1 = np.clip(x0 + 1, 0, W - 1); y1 = np.clip(y0 + 1, 0, H - 1)
    sx = xs - x0; sy = ys - y0
    f00 = field[y0, x0]; f10 = field[y0, x1]
    f01 = field[y1, x0]; f11 = field[y1, x1]
    return (f00 * (1 - sx) * (1 - sy) +
            f10 * sx * (1 - sy) +
            f01 * (1 - sx) * sy +
            f11 * sx * sy)


# ---- Source/Mouth selection with diversity ----
def _farthest_point_sample(coords: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """
    Farthest-point sampling in 2D to spread choices.
    coords: (N,2)
    returns indices of length <=k
    """
    N = coords.shape[0]
    if N == 0 or k <= 0:
        return np.array([], dtype=int)
    idx0 = int(rng.integers(N))
    chosen = [idx0]
    if k == 1:
        return np.array(chosen, dtype=int)
    dist = np.sum((coords - coords[idx0])**2, axis=1)
    for _ in range(1, k):
        nxt = int(np.argmax(dist))
        if dist[nxt] <= 1e-9:
            break
        chosen.append(nxt)
        dist = np.minimum(dist, np.sum((coords - coords[nxt])**2, axis=1))
    return np.array(chosen, dtype=int)


def pick_diverse_sources_and_mouths(points: np.ndarray,
                                    site_vals: np.ndarray,
                                    heightmap: np.ndarray,
                                    sea: float,
                                    mesh: VoronoiMesh,
                                    k_sources: int,
                                    k_mouths: int,
                                    rng: np.random.Generator):
    """
    Sources: land sites from the top 10% elevations, sampled with FPS.
    Mouths : pick diverse coast pixels first, then nearest sites to them (FPS).
    """
    # sources
    land = site_vals > sea
    land_idx = np.where(land)[0]
    if land_idx.size == 0:
        land_idx = np.arange(points.shape[0])
    high = site_vals[land_idx]
    thr = np.quantile(high, 0.90)
    cand_src = land_idx[site_vals[land_idx] >= thr]
    if cand_src.size == 0:
        cand_src = land_idx
    src_coords = points[cand_src]
    src_sel_local = _farthest_point_sample(src_coords, k_sources, rng)
    src_indices = cand_src[src_sel_local]

    # mouths
    coast_mask = find_coast_pixels(heightmap, sea)
    cy, cx = np.where(coast_mask)
    if cx.size == 0:
        # fallback: lowest 10%
        low_thr = np.quantile(site_vals, 0.10)
        cand_mouth = np.where(site_vals <= low_thr)[0]
        if cand_mouth.size == 0:
            cand_mouth = np.arange(points.shape[0])
        mouth_coords = points[cand_mouth]
        mouth_sel_local = _farthest_point_sample(mouth_coords, k_mouths, rng)
        mouth_indices = cand_mouth[mouth_sel_local]
    else:
        # choose diverse coast pixels, then map to nearest sites
        coast_world = np.column_stack([
            cx / (heightmap.shape[1] - 1) * mesh.width_scale,
            cy / (heightmap.shape[0] - 1) * mesh.length_scale
        ])
        coast_sel = _farthest_point_sample(coast_world, k_mouths, rng)
        chosen_coast = coast_world[coast_sel]
        # nearest site to each chosen coast point
        mouth_indices = []
        for cw in chosen_coast:
            d2 = np.sum((points - cw) ** 2, axis=1)
            mouth_indices.append(int(np.argmin(d2)))
        mouth_indices = np.array(mouth_indices, dtype=int)

    return src_indices, mouth_indices


# ---- river walk / path shaping / trimming / carving ----
def trace_river_squig(points: np.ndarray,
                      site_vals: np.ndarray,
                      adj: dict,
                      src_idx: int,
                      mouth_idx: int,
                      bias_downhill: float = 0.70,
                      bias_to_mouth: float = 0.25,
                      jitter_amp: float = 0.05,
                      rng_seed: int = 1337,
                      max_steps: int = 20000):
    rng = np.random.default_rng(rng_seed)
    N = len(points)
    target = points[mouth_idx]
    visited = np.zeros(N, dtype=bool)
    path = [src_idx]
    visited[src_idx] = True

    def forward_progress(a_idx, b_idx):
        a = points[a_idx]; b = points[b_idx]
        da = np.linalg.norm(target - a) + 1e-9
        db = np.linalg.norm(target - b) + 1e-9
        return np.clip((da - db) / (da + 1e-9), 0.0, 1.0)

    cur = src_idx
    iters = 0
    while cur != mouth_idx and iters < max_steps:
        neighbors = [n for n in adj[cur] if not visited[n]]
        if not neighbors:
            neighbors = list(adj[cur])
            if not neighbors:
                break

        e_cur = site_vals[cur]
        scores = []
        for n in neighbors:
            e_n = site_vals[n]
            drop = np.clip(e_cur - e_n, 0.0, None)
            drop_norm = drop / (abs(e_cur) + 1.0)
            prog = forward_progress(cur, n)
            jitter = (_hash_edge(cur, n) / float(1 << 31)) - 0.5
            score = bias_downhill * drop_norm + bias_to_mouth * prog + jitter_amp * jitter
            scores.append((score, n))

        scores.sort(reverse=True, key=lambda x: x[0])
        top = scores[0][0]
        near = [n for sc, n in scores if sc >= top - 1e-6]
        nxt = near[0] if len(near) == 1 else near[int(rng.integers(len(near)))]

        path.append(nxt)
        visited[nxt] = True
        cur = nxt
        iters += 1

    return path


def _normal_of_segment(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    v = p2 - p1
    L = np.linalg.norm(v) + 1e-12
    return np.array([-v[1], v[0]]) / L


def polyline_fractal_subdivide(points: np.ndarray,
                               iters: int,
                               disp0: float,
                               decay: float = 0.6,
                               rng_seed: int = 123) -> np.ndarray:
    rng = np.random.default_rng(rng_seed)
    pts = np.asarray(points, dtype=float)
    disp = float(disp0)
    for _ in range(max(0, int(iters))):
        new_pts = [pts[0]]
        for i in range(len(pts) - 1):
            a = pts[i]; b = pts[i+1]
            m = 0.5 * (a + b)
            n = _normal_of_segment(a, b)
            s = (rng.random() - 0.5) * 2.0
            m = m + n * (disp * s)
            new_pts.append(m)
            new_pts.append(b)
        pts = np.array(new_pts, dtype=float)
        disp *= float(decay)
    return pts


def chaikin_smooth(points: np.ndarray, passes: int = 1) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    for _ in range(max(0, int(passes))):
        if len(pts) < 3:
            return pts
        out = [pts[0]]
        for i in range(len(pts) - 1):
            p = pts[i]; q = pts[i+1]
            Q = 0.75 * p + 0.25 * q
            R = 0.25 * p + 0.75 * q
            out.extend([Q, R])
        out.append(pts[-1])
        pts = np.array(out, dtype=float)
    return pts


def _height_at(world_xy: np.ndarray, heightmap: np.ndarray, mesh: VoronoiMesh) -> float:
    return float(sample_scalar_bilinear(
        heightmap, np.array([world_xy[0]]), np.array([world_xy[1]]),
        mesh.width_scale, mesh.length_scale
    )[0])


def _bisect_to_sea(p_land: np.ndarray,
                   p_water: np.ndarray,
                   sea: float,
                   heightmap: np.ndarray,
                   mesh: VoronoiMesh,
                   max_iter: int = 30,
                   tol_world: float = 1e-4) -> np.ndarray:
    a = p_land.copy()
    b = p_water.copy()
    ha = _height_at(a, heightmap, mesh) - sea
    hb = _height_at(b, heightmap, mesh) - sea
    if ha <= 0 and hb > 0:
        a, b = b, a
        ha, hb = hb, ha
    for _ in range(max_iter):
        m = 0.5 * (a + b)
        hm = _height_at(m, heightmap, mesh) - sea
        if np.linalg.norm(b - a) < tol_world:
            return m
        if hm > 0:
            a, ha = m, hm
        else:
            b, hb = m, hm
    return 0.5 * (a + b)


def trim_path_to_coast(path_xy_world: np.ndarray,
                       heightmap: np.ndarray,
                       sea: float,
                       mesh: VoronoiMesh) -> np.ndarray:
    if len(path_xy_world) < 2:
        return path_xy_world
    trimmed = [path_xy_world[0]]
    prev = path_xy_world[0]
    for i in range(1, len(path_xy_world)):
        cur = path_xy_world[i]
        if _height_at(cur, heightmap, mesh) > sea:
            trimmed.append(cur)
            prev = cur
            continue
        coast_pt = _bisect_to_sea(prev, cur, sea, heightmap, mesh)
        trimmed.append(coast_pt)
        return np.array(trimmed, dtype=float)
    return np.array(trimmed, dtype=float)


def carve_river(heightmap: np.ndarray,
                path_xy_world: np.ndarray,
                mesh: VoronoiMesh,
                width_px: float = 2.0,
                depth: float = 0.06) -> np.ndarray:
    H, W = heightmap.shape
    xs = np.clip(path_xy_world[:, 0] / mesh.width_scale  * (W - 1), 0, W - 1)
    ys = np.clip(path_xy_world[:, 1] / mesh.length_scale * (H - 1), 0, H - 1)
    Y, X = np.mgrid[0:H, 0:W]
    carved = heightmap.copy()
    sigma2 = (width_px ** 2)
    depress = np.zeros_like(heightmap, dtype=float)
    for i in range(len(xs) - 1):
        x1, y1 = xs[i], ys[i]; x2, y2 = xs[i+1], ys[i+1]
        vx, vy = x2 - x1, y2 - y1
        seg2 = vx * vx + vy * vy + 1e-9
        t = ((X - x1) * vx + (Y - y1) * vy) / seg2
        t = np.clip(t, 0.0, 1.0)
        px = x1 + t * vx; py = y1 + t * vy
        d2 = (X - px) ** 2 + (Y - py) ** 2
        brush = np.exp(-d2 / (2.0 * sigma2))
        depress = np.maximum(depress, brush)
    carved = np.clip(carved - depth * depress, 0.0, 1.0)
    return carved


# =========================
# Renderers
# =========================
def render_heatmap_voronoi(
    mesh: VoronoiMesh,
    heightmap: np.ndarray,
    sea: float,
    filename: str = "voronoi.html",
    ocean_deep=(20, 60, 150), ocean_shallow=(150, 200, 230),
    warp_strength: float = 8.0,
    warp_scale: float = 3.0,
    warp_seed: Optional[int] = None,
):
    if warp_strength > 0:
        mesh = warp_mesh(mesh, strength=warp_strength, scale=warp_scale, seed=warp_seed)

    H, W = heightmap.shape
    fig = go.Figure()

    polys = mesh.regions
    centroids = np.array([[p[:, 0].mean(), p[:, 1].mean()] for p in polys])
    i = np.clip((centroids[:, 1] / mesh.length_scale * (H - 1)).astype(int), 0, H - 1)
    j = np.clip((centroids[:, 0] / mesh.width_scale  * (W - 1)).astype(int), 0, W - 1)
    vals = heightmap[i, j]
    colors = _colorize_terrain(vals, sea, ocean_deep=ocean_deep, ocean_shallow=ocean_shallow)

    for poly, col in zip(polys, colors):
        x, y = poly[:, 0], poly[:, 1]
        if not (np.isclose(x[0], x[-1]) and np.isclose(y[0], y[-1])):
            x = np.r_[x, x[0]]; y = np.r_[y, y[0]]
        rgb = f"rgba({int(col[0]*255)},{int(col[1]*255)},{int(col[2]*255)},1)"
        fig.add_trace(go.Scatter(x=x, y=y, fill="toself", fillcolor=rgb,
                                 mode="none", hoverinfo="skip", showlegend=False))

    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(visible=False, range=[0, mesh.width_scale]),
        yaxis=dict(visible=False, range=[0, mesh.length_scale], scaleanchor="x", scaleratio=1),
        paper_bgcolor="white", plot_bgcolor="white"
    )
    html = fig.to_html(full_html=True, include_plotlyjs="cdn")
    html = html.replace("<body>", "<body style='margin:0;'>"
                        "<style>html,body{height:100%;width:100%;margin:0;padding:0;}"
                        ".plotly-container{height:100vh!important;width:100vw!important;}</style>")
    with open(filename, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Wrote: {filename}")


def render_bitmap_html(img_float: np.ndarray,
                       filename: str = "voronoi_bitmap.html",
                       coast_from_height: Optional[np.ndarray] = None,
                       sea: Optional[float] = None,
                       ocean_deep=(20, 60, 150),
                       ocean_shallow=(150, 200, 230),
                       coast_color_main: str = "rgba(25,30,35,0.95)",
                       coast_color_glow: str = "rgba(240,240,240,0.55)",
                       coast_width_main: float = 1.6,
                       coast_width_glow: float = 4.0,
                       river_paths_world: Optional[List[np.ndarray]] = None,
                       mesh_for_rivers: Optional[VoronoiMesh] = None,
                       river_width_main: float = 2.4,
                       river_width_glow: float = 5.2):
    H, W, C = img_float.shape
    assert C == 3, "Expected HxWx3 image"
    img_uint8 = np.clip(img_float * 255.0 + 0.5, 0, 255).astype(np.uint8)

    os = np.asarray(ocean_shallow, dtype=float) / 255.0
    river_rgb = os
    river_color_main = f"rgba({int(river_rgb[0]*255)},{int(river_rgb[1]*255)},{int(river_rgb[2]*255)},0.98)"
    river_color_glow = f"rgba({int(river_rgb[0]*255)},{int(river_rgb[1]*255)},{int(river_rgb[2]*255)},0.55)"

    fig = go.Figure(go.Image(z=img_uint8))
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(visible=False, range=[0, W]),
        yaxis=dict(visible=False, range=[0, H], scaleanchor="x", scaleratio=1, autorange="reversed"),
        paper_bgcolor="white", plot_bgcolor="white"
    )
    fig.update_xaxes(constrain="domain", fixedrange=True)
    fig.update_yaxes(scaleanchor="x", scaleratio=1, fixedrange=True)

    if coast_from_height is not None and sea is not None:
        cx, cy = marching_squares_segments(coast_from_height, iso=sea)
        if coast_width_glow > 0:
            fig.add_trace(go.Scatter(
                x=cx, y=cy, mode="lines",
                line=dict(width=coast_width_glow, color=coast_color_glow, shape="linear"),
                hoverinfo="skip", showlegend=False
            ))
        fig.add_trace(go.Scatter(
            x=cx, y=cy, mode="lines",
            line=dict(width=coast_width_main, color=coast_color_main, shape="linear"),
            hoverinfo="skip", showlegend=False
        ))

    if river_paths_world and mesh_for_rivers is not None:
        H_img, W_img = fig.data[0].z.shape[:2]
        for path in river_paths_world:
            xs = np.clip(path[:, 0] / mesh_for_rivers.width_scale  * (W_img - 1), 0, W_img - 1)
            ys = np.clip(path[:, 1] / mesh_for_rivers.length_scale * (H_img - 1), 0, H_img - 1)
            if river_width_glow > 0:
                fig.add_trace(go.Scatter(
                    x=xs, y=ys, mode="lines",
                    line=dict(width=river_width_glow, color=river_color_glow, shape="spline", smoothing=1.0),
                    hoverinfo="skip", showlegend=False
                ))
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode="lines",
                line=dict(width=river_width_main, color=river_color_main, shape="spline", smoothing=1.0),
                hoverinfo="skip", showlegend=False
            ))

    html = fig.to_html(full_html=True, include_plotlyjs="cdn")
    html = html.replace("<body>", "<body style='margin:0;'>"
                        "<style>html,body{height:100%;width:100%;margin:0;padding:0;}"
                        ".plotly-container{height:100vh!important;width:100vw!important;}</style>")
    with open(filename, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Wrote: {filename}")


# =========================
# Minimal tests
# =========================
def _self_test_colormap():
    sea = 0.6
    vals = np.linspace(0, 1, 11)
    colors = _colorize_terrain(vals, sea)
    assert colors.shape == (11, 3)
    assert np.all(colors >= 0) and np.all(colors <= 1)
    i_sea = np.searchsorted(vals, sea)
    if 0 < i_sea < len(vals):
        assert not np.allclose(colors[i_sea-1], colors[i_sea])

def _self_test_heightmap():
    h, sea = generate_heightmap(mode="continents", width_px=128, length_px=64, water_pct=0.65, seed=123)
    assert h.shape == (64, 128)
    assert 0.0 <= sea <= 1.0
    frac_water = (h <= sea).mean()
    assert 0.50 <= frac_water <= 0.80, f"Water fraction {frac_water:.2f} out of bounds"

def _self_test_warp():
    mesh = build_voronoi(num_points=1000, width=800, length=400, seed=7)
    warped = warp_mesh(mesh, strength=6.0, scale=4.0, seed=99)
    assert len(mesh.regions) == len(warped.regions)
    for a, b in zip(mesh.regions, warped.regions):
        assert a.shape == b.shape


# =========================
# River generation wrapper (diverse starts/ends, count knob)
# =========================
def generate_rivers(mesh: VoronoiMesh,
                    height: np.ndarray,
                    sea: float,
                    count: int,
                    min_len_world: float,
                    max_len_world: float,
                    meander: float = 0.6,
                    fractal_iters: int = 0,
                    fractal_displace: float = 5.0,
                    fractal_decay: float = 0.6,
                    smooth_passes: int = 0,
                    max_tries: int = 80,
                    rng_seed: int = 12345) -> List[np.ndarray]:
    rng = np.random.default_rng(rng_seed)
    adj = delaunay_adjacency_from_points(mesh.points)
    site_vals = sample_scalar_bilinear(
        height, mesh.points[:, 0], mesh.points[:, 1],
        mesh.width_scale, mesh.length_scale
    )

    # Convert meander to scoring weights:
    meander = float(np.clip(meander, 0.0, 1.0))
    bias_downhill = 0.70
    bias_to_mouth = np.interp(meander, [0, 1], [0.35, 0.10])
    jitter_amp   = np.interp(meander, [0, 1], [0.02, 0.10])

    # pick multiple diverse sources & mouths up-front
    kS = max(1, min(count * 2, 12))
    kM = max(1, min(count * 3, 18))
    srcCandidates, mouthCandidates = pick_diverse_sources_and_mouths(
        mesh.points, site_vals, height, sea, mesh, kS, kM, rng
    )

    rivers: List[np.ndarray] = []
    used_pairs = set()
    tries = 0

    while len(rivers) < count and tries < max_tries:
        tries += 1
        if srcCandidates.size == 0 or mouthCandidates.size == 0:
            break
        src_idx = int(rng.choice(srcCandidates))
        mouth_idx = int(rng.choice(mouthCandidates))
        if (src_idx, mouth_idx) in used_pairs or src_idx == mouth_idx:
            continue

        path_nodes = trace_river_squig(
            mesh.points, site_vals, adj,
            src_idx, mouth_idx,
            bias_downhill=bias_downhill,
            bias_to_mouth=bias_to_mouth,
            jitter_amp=jitter_amp,
            rng_seed=int(rng.integers(1 << 31))
        )
        if len(path_nodes) < 2:
            continue

        river_xy = mesh.points[np.array(path_nodes, dtype=int)]

        # Apply fractal jaggedness BEFORE trimming
        if fractal_iters > 0 and fractal_displace > 0:
            river_xy = polyline_fractal_subdivide(
                river_xy, iters=int(fractal_iters),
                disp0=float(fractal_displace),
                decay=float(fractal_decay),
                rng_seed=int(rng.integers(1 << 31))
            )
        if smooth_passes > 0:
            river_xy = chaikin_smooth(river_xy, passes=int(smooth_passes))

        # Trim to coast
        river_xy = trim_path_to_coast(river_xy, height, sea, mesh)
        if len(river_xy) < 2:
            continue

        # Length constraints AFTER trim
        L = float(np.sum(np.linalg.norm(river_xy[1:] - river_xy[:-1], axis=1)))
        if L < min_len_world or L > max_len_world:
            continue

        rivers.append(river_xy)
        used_pairs.add((src_idx, mouth_idx))

        # Make next mouths more diverse by removing nearby picks
        mouth_pt = mesh.points[mouth_idx]
        d2 = np.sum((mesh.points[mouthCandidates] - mouth_pt) ** 2, axis=1)
        far_mask = d2 > (0.08 * max(mesh.width_scale, mesh.length_scale)) ** 2
        if np.any(far_mask):
            mouthCandidates = mouthCandidates[far_mask]

    return rivers


# =========================
# Example usage
# =========================
if __name__ == "__main__":
    # Run light tests
    _self_test_colormap()
    _self_test_heightmap()
    _self_test_warp()

    WIDTH, LENGTH = 1600.0, 900.0

    # 1) Voronoi sites
    mesh = build_voronoi(num_points=8000, width=WIDTH, length=LENGTH, seed=42)

    # 2) Heightmap (choose mode)
    height, sea = generate_heightmap(
        mode="continents",         # "world" | "continents" | "archipelago" | "islands"
        width_px=1024,
        length_px=512,
        water_pct=0.67,
        seed=None,                 # None => new world each run
        lakes_pct=0.03,
        lakes_scale=0.9,
        lakes_smooth=5,
        mountain_strength=0.55,
        mountain_gamma=1.8,
        mountain_mask_scale=3.5,
        max_land_elev=1.00,
        # Terrain tamers (tune these!)
        vignette_strength=0.38,      # push edges downward a bit
        vignette_margin=0.10,        # how wide the edge falloff is
        continent_bias_strength=0.50,# gently bias mass toward middle
        recentre_landmass=True       # roll map so land centroid is centered
    )

    # -------- Feature knobs --------
    RENDER_MODE = "bitmap"        # "bitmap" or "polygons"
    RIVERS = True                 # enable river tracing/overlay
    DO_CARVE = True               # depress height along each river before colorizing

    # River count and length constraints (world units)
    RIVER_COUNT = 5               # <— knob for number of rivers
    RIVER_MIN_LEN_WORLD = WIDTH * 0.18
    RIVER_MAX_LEN_WORLD = WIDTH * 0.80
    RIVER_MAX_TRIES = 120

    # Meander + jaggedness knobs
    RIVER_MEANDER = 0.65            # 0..1, higher = more wandering
    RIVER_FRACTAL_ITERS = 2         # midpoint-fractal passes (0 = off)
    RIVER_FRACTAL_DISPLACE = WIDTH * 0.01
    RIVER_FRACTAL_DECAY = 0.6
    RIVER_SMOOTH_PASSES = 1

    river_paths: List[np.ndarray] = []
    if RIVERS:
        river_paths = generate_rivers(
            mesh, height, sea,
            count=RIVER_COUNT,
            min_len_world=RIVER_MIN_LEN_WORLD,
            max_len_world=RIVER_MAX_LEN_WORLD,
            meander=RIVER_MEANDER,
            fractal_iters=RIVER_FRACTAL_ITERS,
            fractal_displace=RIVER_FRACTAL_DISPLACE,
            fractal_decay=RIVER_FRACTAL_DECAY,
            smooth_passes=RIVER_SMOOTH_PASSES,
            max_tries=RIVER_MAX_TRIES,
            rng_seed=777
        )

        if DO_CARVE and len(river_paths) > 0:
            for river_xy_world in river_paths:
                height = carve_river(height, river_xy_world, mesh,
                                     width_px=2.7, depth=0.07)
            # re-normalize and recompute sea to keep water % consistent
            height -= height.min()
            height /= height.max() + 1e-12
            sea = float(np.quantile(height, 0.67))

    if RENDER_MODE == "bitmap":
        img = colorize_bitmap(height, sea)
        img_blurred = _box_blur_rgb(img, k=3)
        render_bitmap_html(
            img_blurred,
            filename="voronoi_bitmap.html",
            coast_from_height=height,
            sea=sea,
            ocean_deep=(20, 60, 150),
            ocean_shallow=(150, 200, 230),
            coast_color_main="rgba(20,25,30,0.95)",
            coast_color_glow="rgba(250,250,250,0.55)",
            coast_width_main=1.6,
            coast_width_glow=4.2,
            river_paths_world=river_paths if RIVERS else None,
            mesh_for_rivers=mesh,
            river_width_main=2.6,
            river_width_glow=5.4
        )
    else:
        render_heatmap_voronoi(
            mesh, height, sea,
            filename="voronoi.html",
            warp_strength=8.0,
            warp_scale=3.0,
            warp_seed=None,
        )
