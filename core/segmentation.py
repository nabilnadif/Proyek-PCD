"""
Modul core/segmentation.py
Implementasi fitur Menu Segmentation (M9 & M12-M13):
- Global (Otsu) & Adaptive Thresholding
- Contour Detection (ekstraksi batas wilayah objek)
- Hough Transform (Line & Circle)
- Thinning (Skeletonization Zhang-Suen)
"""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from .utils import ensure_float64, to_grayscale_float, clip_uint8, convolve2d_single
from .edge_detection import apply_canny


# =====================================================================
# 1. GLOBAL (OTSU) & ADAPTIVE THRESHOLDING
# =====================================================================
def compute_otsu_threshold(gray_u8: np.ndarray) -> int:
    """Menghitung nilai ambang optimal metode Otsu (maksimasi between-class variance)."""
    hist, _ = np.histogram(gray_u8.ravel(), bins=256, range=(0, 256))
    total = gray_u8.size
    if total == 0:
        return 128

    prob = hist.astype(np.float64) / total
    omega = np.cumsum(prob)
    mu = np.cumsum(prob * np.arange(256, dtype=np.float64))
    mu_total = mu[-1]

    denom = omega * (1.0 - omega)
    valid = denom > 1e-12
    sigma_b_sq = np.zeros(256, dtype=np.float64)
    sigma_b_sq[valid] = ((mu_total * omega[valid] - mu[valid]) ** 2) / denom[valid]
    return int(np.argmax(sigma_b_sq))


def apply_global_otsu_threshold(img_arr: np.ndarray) -> np.ndarray:
    """Segmentasi Global Thresholding otomatis menggunakan metode Otsu."""
    gray = clip_uint8(to_grayscale_float(img_arr))
    t_opt = compute_otsu_threshold(gray)
    return np.where(gray >= t_opt, 255, 0).astype(np.uint8)


def apply_adaptive_threshold(
    img_arr: np.ndarray, block_size: int = 15, c_const: float = 5.0
) -> np.ndarray:
    """
    Adaptive Mean Thresholding:
    Piksel = 255 jika f(x,y) > mean_lokal(block_size) - C, else 0.
    """
    bs = int(block_size)
    if bs < 3:
        bs = 3
    if bs % 2 == 0:
        bs += 1

    gray = to_grayscale_float(img_arr)
    kernel = np.ones((bs, bs), dtype=np.float64) / (bs * bs)
    local_mean = convolve2d_single(gray, kernel, flip_kernel=False)
    out = np.where(gray > (local_mean - float(c_const)), 255, 0)
    return out.astype(np.uint8)


# =====================================================================
# 2. CONTOUR DETECTION
# =====================================================================
def apply_contour_detection(
    img_arr: np.ndarray, threshold: int = None, overlay: bool = True
) -> np.ndarray:
    """
    Deteksi kontur / batas wilayah objek menggunakan ekstraksi batas morfologis:
    Boundary(A) = A - Erode(A, B_3x3).
    Jika overlay=True, kontur digambar berwarna hijau terang di atas citra asli.
    """
    gray = clip_uint8(to_grayscale_float(img_arr))
    t = compute_otsu_threshold(gray) if threshold is None else int(threshold)
    binary = (gray >= t).astype(np.uint8)

    padded = np.pad(binary, ((1, 1), (1, 1)), mode="constant", constant_values=0)
    windows = sliding_window_view(padded, (3, 3))
    eroded = np.min(windows, axis=(-2, -1))
    contour_mask = (binary - eroded) > 0

    if not overlay:
        return (contour_mask.astype(np.uint8) * 255)

    arr = ensure_float64(img_arr)
    if arr.ndim == 2:
        vis = np.stack([arr, arr, arr], axis=-1)
    else:
        vis = arr[:, :, :3].copy()

    vis[contour_mask] = [0.0, 255.0, 80.0]
    return clip_uint8(vis)


# =====================================================================
# 3. HOUGH TRANSFORM (LINE & CIRCLE)
# =====================================================================
def _draw_line_rgb(canvas: np.ndarray, rho: float, theta: float, color=(255, 40, 40)):
    h, w = canvas.shape[:2]
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    if abs(sin_t) > 1e-6:
        xs = np.arange(w)
        ys = np.rint((rho - xs * cos_t) / sin_t).astype(np.int64)
        valid = (ys >= 0) & (ys < h)
        canvas[ys[valid], xs[valid]] = color
    if abs(cos_t) > 1e-6:
        ys = np.arange(h)
        xs = np.rint((rho - ys * sin_t) / cos_t).astype(np.int64)
        valid = (xs >= 0) & (xs < w)
        canvas[ys[valid], xs[valid]] = color


def apply_hough_line(
    img_arr: np.ndarray, num_lines: int = 10, threshold_ratio: float = 0.45
) -> np.ndarray:
    """
    Hough Line Transform (rho = x*cos(theta) + y*sin(theta)):
    Mendeteksi garis dominan dari peta tepi Canny dan menggambarnya di atas citra.
    """
    arr = ensure_float64(img_arr)
    vis = np.stack([arr, arr, arr], axis=-1) if arr.ndim == 2 else arr[:, :, :3].copy()
    h, w = vis.shape[:2]

    edges = apply_canny(img_arr, sigma=1.4, upper_threshold=100.0, lower_threshold=40.0)
    ys, xs = np.nonzero(edges)
    if ys.size == 0:
        return clip_uint8(vis)

    thetas = np.deg2rad(np.arange(-90.0, 90.0, 1.0))
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)

    diag = int(np.ceil(np.hypot(h, w)))
    num_rhos = 2 * diag + 1
    accumulator = np.zeros((num_rhos, len(thetas)), dtype=np.int32)

    # Voting vectorized per sudut theta (hemat memori dan cepat)
    for idx in range(len(thetas)):
        rhos = np.rint(xs * cos_t[idx] + ys * sin_t[idx]).astype(np.int64) + diag
        counts = np.bincount(rhos, minlength=num_rhos)
        accumulator[:, idx] = counts[:num_rhos]

    max_vote = accumulator.max()
    if max_vote == 0:
        return clip_uint8(vis)

    min_votes = max(15, int(max_vote * float(threshold_ratio)))

    # Non-maximum suppression sederhana di ruang accumulator untuk memilih garis terkuat
    acc_copy = accumulator.copy()
    drawn = 0
    for _ in range(max(1, int(num_lines))):
        flat_idx = int(np.argmax(acc_copy))
        votes = acc_copy.ravel()[flat_idx]
        if votes < min_votes:
            break
        r_idx, t_idx = np.unravel_index(flat_idx, acc_copy.shape)
        rho = float(r_idx - diag)
        theta = float(thetas[t_idx])
        _draw_line_rgb(vis, rho, theta, color=(255, 40, 40))
        drawn += 1

        # Tekan lingkungan sekitar puncak agar tidak duplikat garis yang sama
        r0, r1 = max(0, r_idx - 12), min(num_rhos, r_idx + 13)
        t0, t1 = max(0, t_idx - 8), min(len(thetas), t_idx + 9)
        acc_copy[r0:r1, t0:t1] = 0

    return clip_uint8(vis)


def apply_hough_circle(
    img_arr: np.ndarray,
    r_min: int = 15,
    r_max: int = 60,
    num_circles: int = 5,
) -> np.ndarray:
    """
    Hough Circle Transform (a = x - r*cos(t), b = y - r*sin(t)):
    Mendeteksi lingkaran pada rentang jari-jari [r_min, r_max] dan menggambarnya di atas citra.
    """
    arr = ensure_float64(img_arr)
    vis = np.stack([arr, arr, arr], axis=-1) if arr.ndim == 2 else arr[:, :, :3].copy()
    h, w = vis.shape[:2]

    r_lo = max(5, int(r_min))
    r_hi = max(r_lo + 1, min(int(r_max), min(h, w) // 2))

    # Downsample ringan bila citra sangat besar agar tetap responsif
    scale = 1.0
    if max(h, w) > 360:
        scale = 360.0 / max(h, w)
        from PIL import Image
        small = np.asarray(
            Image.fromarray(clip_uint8(to_grayscale_float(img_arr))).resize(
                (max(16, int(w * scale)), max(16, int(h * scale))),
                Image.Resampling.BILINEAR,
            ),
            dtype=np.float64,
        )
    else:
        small = to_grayscale_float(img_arr)

    sh, sw = small.shape
    edges = apply_canny(small, sigma=1.2, upper_threshold=90.0, lower_threshold=35.0)
    ys, xs = np.nonzero(edges)
    if ys.size == 0:
        return clip_uint8(vis)

    sr_min = max(4, int(round(r_lo * scale)))
    sr_max = max(sr_min + 1, int(round(r_hi * scale)))
    r_step = max(1, (sr_max - sr_min) // 25)
    radii = list(range(sr_min, sr_max + 1, r_step))

    angles = np.deg2rad(np.arange(0, 360, 10))
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)

    candidates = []
    for r in radii:
        acc = np.zeros((sh, sw), dtype=np.int32)
        for ca, sa in zip(cos_a, sin_a):
            a = np.rint(xs - r * ca).astype(np.int64)
            b = np.rint(ys - r * sa).astype(np.int64)
            valid = (a >= 0) & (a < sw) & (b >= 0) & (b < sh)
            np.add.at(acc, (b[valid], a[valid]), 1)
        best_idx = int(np.argmax(acc))
        best_val = int(acc.ravel()[best_idx])
        # Normalisasi skor terhadap jumlah sudut voting (36)
        score = best_val / float(len(angles))
        if score >= 0.40:
            by, bx = np.unravel_index(best_idx, acc.shape)
            candidates.append((score, bx / scale, by / scale, r / scale))

    candidates.sort(key=lambda item: item[0], reverse=True)

    selected = []
    for score, cx, cy, rad in candidates:
        if all(np.hypot(cx - sx, cy - sy) > max(12.0, 0.5 * min(rad, sr)) for _, sx, sy, sr in selected):
            selected.append((score, cx, cy, rad))
            if len(selected) >= max(1, int(num_circles)):
                break

    draw_angles = np.linspace(0, 2 * np.pi, 360, endpoint=False)
    for _, cx, cy, rad in selected:
        px = np.rint(cx + rad * np.cos(draw_angles)).astype(np.int64)
        py = np.rint(cy + rad * np.sin(draw_angles)).astype(np.int64)
        valid = (px >= 0) & (px < w) & (py >= 0) & (py < h)
        vis[py[valid], px[valid]] = [0.0, 255.0, 255.0]
        # Tandai titik pusat lingkaran
        icx, icy = int(round(cx)), int(round(cy))
        if 1 <= icx < w - 1 and 1 <= icy < h - 1:
            vis[icy - 1 : icy + 2, icx - 1 : icx + 2] = [255.0, 50.0, 50.0]

    return clip_uint8(vis)


# =====================================================================
# 4. THINNING (SKELETONIZATION - ZHANG-SUEN)
# =====================================================================
def apply_thinning(img_arr: np.ndarray, max_iter: int = 80, invert_fg: bool = False) -> np.ndarray:
    """
    Skeletonization (Penipisan objek biner) menggunakan algoritma Zhang-Suen.
    Citra otomatis di-threshold dengan Otsu terlebih dahulu jika belum biner.
    """
    gray = clip_uint8(to_grayscale_float(img_arr))
    t = compute_otsu_threshold(gray)
    if invert_fg:
        img = (gray < t).astype(np.uint8)
    else:
        img = (gray >= t).astype(np.uint8)

    def _zhang_suen_step(mat: np.ndarray, step: int) -> np.ndarray:
        p2 = mat[:-2, 1:-1]
        p3 = mat[:-2, 2:]
        p4 = mat[1:-1, 2:]
        p5 = mat[2:, 2:]
        p6 = mat[2:, 1:-1]
        p7 = mat[2:, :-2]
        p8 = mat[1:-1, :-2]
        p9 = mat[:-2, :-2]

        neighbors = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
        transitions = (
            ((p2 == 0) & (p3 == 1)).astype(np.uint8)
            + ((p3 == 0) & (p4 == 1)).astype(np.uint8)
            + ((p4 == 0) & (p5 == 1)).astype(np.uint8)
            + ((p5 == 0) & (p6 == 1)).astype(np.uint8)
            + ((p6 == 0) & (p7 == 1)).astype(np.uint8)
            + ((p7 == 0) & (p8 == 1)).astype(np.uint8)
            + ((p8 == 0) & (p9 == 1)).astype(np.uint8)
            + ((p9 == 0) & (p2 == 1)).astype(np.uint8)
        )

        if step == 1:
            c1 = (p2 * p4 * p6) == 0
            c2 = (p4 * p6 * p8) == 0
        else:
            c1 = (p2 * p4 * p8) == 0
            c2 = (p2 * p6 * p8) == 0

        cond = (
            (mat[1:-1, 1:-1] == 1)
            & (neighbors >= 2)
            & (neighbors <= 6)
            & (transitions == 1)
            & c1
            & c2
        )
        out = np.zeros_like(mat, dtype=bool)
        out[1:-1, 1:-1] = cond
        return out

    padded = np.pad(img, ((1, 1), (1, 1)), mode="constant", constant_values=0)
    for _ in range(int(max_iter)):
        m1 = _zhang_suen_step(padded, 1)
        padded[m1] = 0
        m2 = _zhang_suen_step(padded, 2)
        padded[m2] = 0
        if not (m1.any() or m2.any()):
            break

    skeleton = padded[1:-1, 1:-1]
    return (skeleton * 255).astype(np.uint8)
