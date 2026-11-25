"""
Modul core/edge_detection.py
Implementasi deteksi tepi (Slide M10 & M11):
- 1'st Differential Gradient: Sobel, Prewitt, Robert
- 2'nd Differential Gradient: Laplacian, Laplacian of Gaussian (LoG), Canny
- Compass (8 arah mata angin Kirsch)
Seluruh citra input otomatis dikonversi ke grayscale sebelum deteksi tepi.
"""
from collections import deque
import numpy as np
from .utils import to_grayscale_float, clip_uint8, convolve2d_single


def _normalize_magnitude(mag: np.ndarray, normalize: bool = False) -> np.ndarray:
    if normalize:
        m_max = mag.max()
        if m_max > 1e-12:
            mag = (mag / m_max) * 255.0
    return clip_uint8(mag)


# =====================================================================
# 1. FIRST DIFFERENTIAL GRADIENT (Sobel, Prewitt, Robert)
# =====================================================================
def apply_sobel(img_arr: np.ndarray, normalize: bool = False) -> np.ndarray:
    """Operator Sobel: M = sqrt(Sx^2 + Sy^2)."""
    gray = to_grayscale_float(img_arr)
    sx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
    sy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float64)
    gx = convolve2d_single(gray, sx, flip_kernel=False)
    gy = convolve2d_single(gray, sy, flip_kernel=False)
    mag = np.hypot(gx, gy)
    return _normalize_magnitude(mag, normalize=normalize)


def apply_prewitt(img_arr: np.ndarray, normalize: bool = False) -> np.ndarray:
    """Operator Prewitt: M = sqrt(Px^2 + Py^2)."""
    gray = to_grayscale_float(img_arr)
    px = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float64)
    py = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=np.float64)
    gx = convolve2d_single(gray, px, flip_kernel=False)
    gy = convolve2d_single(gray, py, flip_kernel=False)
    mag = np.hypot(gx, gy)
    return _normalize_magnitude(mag, normalize=normalize)


def apply_robert(img_arr: np.ndarray, normalize: bool = False) -> np.ndarray:
    """Operator Robert Cross 2x2: M = sqrt(R+^2 + R-^2)."""
    gray = to_grayscale_float(img_arr)
    r_plus = np.array([[1, 0], [0, -1]], dtype=np.float64)
    r_minus = np.array([[0, 1], [-1, 0]], dtype=np.float64)
    g1 = convolve2d_single(gray, r_plus, flip_kernel=False)
    g2 = convolve2d_single(gray, r_minus, flip_kernel=False)
    mag = np.hypot(g1, g2)
    return _normalize_magnitude(mag, normalize=normalize)


# =====================================================================
# 2. SECOND DIFFERENTIAL GRADIENT (Laplacian, LoG, Canny)
# =====================================================================
def apply_laplacian(img_arr: np.ndarray, neighbor_mode: str = "4") -> np.ndarray:
    """
    Operator Laplacian:
    - '4': [[0, 1, 0], [1, -4, 1], [0, 1, 0]]
    - '8': [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]
    """
    gray = to_grayscale_float(img_arr)
    if str(neighbor_mode) == "8":
        kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=np.float64)
    else:
        kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)
    res = np.abs(convolve2d_single(gray, kernel, flip_kernel=True))
    return clip_uint8(res)


def _gaussian_kernel_2d(sigma: float, size: int = None) -> np.ndarray:
    s = max(0.3, float(sigma))
    if size is None:
        size = int(2 * round(3 * s) + 1)
    if size % 2 == 0:
        size += 1
    half = size // 2
    ax = np.arange(-half, half + 1, dtype=np.float64)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * s ** 2))
    return kernel / np.sum(kernel)


def apply_log(img_arr: np.ndarray, mode: str = "mexican_hat", sigma: float = 1.4) -> np.ndarray:
    """
    Laplacian of Gaussian (LoG):
    - 'mexican_hat': Konvolusi langsung kernel Mexican Hat 5x5.
    - 'gaussian_laplacian': Gaussian smoothing (sigma) diikuti Laplacian 8-tetangga.
    """
    gray = to_grayscale_float(img_arr)
    if mode == "gaussian_laplacian":
        g_kernel = _gaussian_kernel_2d(sigma)
        smoothed = convolve2d_single(gray, g_kernel, flip_kernel=False)
        lap_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=np.float64)
        res = np.abs(convolve2d_single(smoothed, lap_kernel, flip_kernel=True))
    else:
        mexican_hat = np.array(
            [
                [0, 0, -1, 0, 0],
                [0, -1, -2, -1, 0],
                [-1, -2, 16, -2, -1],
                [0, -1, -2, -1, 0],
                [0, 0, -1, 0, 0],
            ],
            dtype=np.float64,
        )
        res = np.abs(convolve2d_single(gray, mexican_hat, flip_kernel=True))
    return clip_uint8(res)


def apply_canny(
    img_arr: np.ndarray,
    sigma: float = 1.4,
    upper_threshold: float = 100.0,
    lower_threshold: float = 40.0,
) -> np.ndarray:
    """
    Algoritma Canny Edge Detection (4 tahap lengkap):
    1. Gaussian Smoothing (SD = sigma)
    2. Gradien Sobel (Magnitudo & Arah)
    3. Non-Maximal Suppression (NMS)
    4. Double Thresholding & Hysteresis Edge Tracking (T1=upper, T2=lower)
    """
    t_high = float(upper_threshold)
    t_low = float(lower_threshold)
    if t_low > t_high:
        t_low, t_high = t_high, t_low

    gray = to_grayscale_float(img_arr)

    # Tahap 1: Gaussian Smoothing
    g_kernel = _gaussian_kernel_2d(sigma)
    smoothed = convolve2d_single(gray, g_kernel, flip_kernel=False)

    # Tahap 2: Gradien Sobel
    sx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
    sy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float64)
    gx = convolve2d_single(smoothed, sx, flip_kernel=False)
    gy = convolve2d_single(smoothed, sy, flip_kernel=False)
    mag = np.hypot(gx, gy)
    angle = np.rad2deg(np.arctan2(gy, gx)) % 180.0

    # Tahap 3: Non-Maximal Suppression (vectorized)
    h, w = mag.shape
    nms = np.zeros_like(mag)
    if h >= 3 and w >= 3:
        center = mag[1:-1, 1:-1]
        ang = angle[1:-1, 1:-1]

        # 0 derajat (horizontal): kiri & kanan
        m0 = ((ang < 22.5) | (ang >= 157.5))
        q0 = mag[1:-1, 2:]
        r0 = mag[1:-1, :-2]

        # 45 derajat (diagonal /): kanan-atas & kiri-bawah
        m45 = (ang >= 22.5) & (ang < 67.5)
        q45 = mag[:-2, 2:]
        r45 = mag[2:, :-2]

        # 90 derajat (vertikal): atas & bawah
        m90 = (ang >= 67.5) & (ang < 112.5)
        q90 = mag[:-2, 1:-1]
        r90 = mag[2:, 1:-1]

        # 135 derajat (diagonal \): kiri-atas & kanan-bawah
        m135 = (ang >= 112.5) & (ang < 157.5)
        q135 = mag[:-2, :-2]
        r135 = mag[2:, 2:]

        keep = (
            (m0 & (center >= q0) & (center >= r0))
            | (m45 & (center >= q45) & (center >= r45))
            | (m90 & (center >= q90) & (center >= r90))
            | (m135 & (center >= q135) & (center >= r135))
        )
        nms[1:-1, 1:-1] = np.where(keep, center, 0.0)

    # Tahap 4: Double Threshold & Hysteresis Tracking (BFS)
    strong = nms >= t_high
    weak = (nms >= t_low) & (~strong)

    edges = np.zeros((h, w), dtype=np.uint8)
    edges[strong] = 255

    queue = deque(zip(*np.nonzero(strong)))
    while queue:
        r, c = queue.popleft()
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and weak[nr, nc] and edges[nr, nc] == 0:
                    edges[nr, nc] = 255
                    queue.append((nr, nc))

    return edges


# =====================================================================
# 3. COMPASS EDGE DETECTION (8 Arah Mata Angin - Kirsch)
# =====================================================================
def apply_compass(img_arr: np.ndarray) -> np.ndarray:
    """
    Compass Edge Detection menggunakan 8 mask arah mata angin Kirsch
    (U, TL, T, Tg, S, BD, B, BL), lalu mengambil nilai maksimum: G = max_i{G_i}.
    """
    gray = to_grayscale_float(img_arr)

    masks = [
        # Utara (N)
        np.array([[5, 5, 5], [-3, 0, -3], [-3, -3, -3]], dtype=np.float64),
        # Timur Laut (NE)
        np.array([[-3, 5, 5], [-3, 0, 5], [-3, -3, -3]], dtype=np.float64),
        # Timur (E)
        np.array([[-3, -3, 5], [-3, 0, 5], [-3, -3, 5]], dtype=np.float64),
        # Tenggara (SE)
        np.array([[-3, -3, -3], [-3, 0, 5], [-3, 5, 5]], dtype=np.float64),
        # Selatan (S)
        np.array([[-3, -3, -3], [-3, 0, -3], [5, 5, 5]], dtype=np.float64),
        # Barat Daya (SW)
        np.array([[-3, -3, -3], [5, 0, -3], [5, 5, -3]], dtype=np.float64),
        # Barat (W)
        np.array([[5, -3, -3], [5, 0, -3], [5, -3, -3]], dtype=np.float64),
        # Barat Laut (NW)
        np.array([[5, 5, -3], [5, 0, -3], [-3, -3, -3]], dtype=np.float64),
    ]

    responses = [convolve2d_single(gray, m, flip_kernel=False) for m in masks]
    max_resp = np.max(np.stack(responses, axis=0), axis=0)
    # Normalisasi skala Kirsch (faktor pembagi ~15) agar tajam tanpa jenuh total
    max_resp = np.maximum(max_resp, 0.0) / 5.0
    return clip_uint8(max_resp)
