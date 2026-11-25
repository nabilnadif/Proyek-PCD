"""
Modul core/enhancement.py
Implementasi perbaikan kualitas citra (Image Enhancement):
Brightness, Contrast (Scaling & Stretching), Histogram Equalization,
Smoothing (Spatial: Lowpass, Median; Frequency: ILPF, BLPF),
Sharpening (Spatial: Highpass, Highboost; Frequency: IHPF, BHPF),
dan Geometrics Correction.
"""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from .utils import (
    ensure_float64,
    clip_uint8,
    convolve2d,
    apply_per_channel,
)
from .basic_ops import _bilinear_sample


# =====================================================================
# 1. BRIGHTNESS
# =====================================================================
def apply_brightness(img_arr: np.ndarray, k: float) -> np.ndarray:
    """Penyesuaian kecerahan: f_o(x,y) = f_i(x,y) + k, clipping [0, 255]."""
    arr = ensure_float64(img_arr)
    return clip_uint8(arr + float(k))


# =====================================================================
# 2. CONTRAST (Scaling & Stretching)
# =====================================================================
def apply_contrast_scaling(img_arr: np.ndarray, gain: float = 1.5, pivot: float = 128.0) -> np.ndarray:
    """Contrast Scaling: f_o = G * (f_i - P) + P."""
    arr = ensure_float64(img_arr)
    res = float(gain) * (arr - float(pivot)) + float(pivot)
    return clip_uint8(res)


def apply_contrast_stretching(
    img_arr: np.ndarray,
    r_min: float = None,
    r_max: float = None,
) -> np.ndarray:
    """
    Contrast Stretching: s = ((r - r_min) / (r_max - r_min)) * 255.
    Jika r_min atau r_max tidak diisi, otomatis mengambil nilai min/max aktual citra.
    """
    arr = ensure_float64(img_arr)
    lo = float(arr.min()) if r_min is None else float(r_min)
    hi = float(arr.max()) if r_max is None else float(r_max)

    if hi <= lo:
        raise ValueError("Nilai r_max harus lebih besar dari r_min.")

    res = ((arr - lo) / (hi - lo)) * 255.0
    return clip_uint8(res)


# =====================================================================
# 3. HISTOGRAM EQUALIZATION
# =====================================================================
def _equalize_channel(ch: np.ndarray) -> np.ndarray:
    ch_u8 = clip_uint8(ch)
    hist, _ = np.histogram(ch_u8.ravel(), bins=256, range=(0, 256))
    pdf = hist.astype(np.float64) / ch_u8.size
    cdf = np.cumsum(pdf)
    lut = np.rint(cdf * 255.0)
    return lut[ch_u8]


def apply_histogram_equalization(img_arr: np.ndarray) -> np.ndarray:
    """
    Histogram Equalization:
    PDF P_r(r_k) = n_k / n, CDF s_k = sum(P_r(r_j)), dipetakan ke [0, 255].
    Mendukung citra grayscale maupun RGB (per kanal).
    """
    res = apply_per_channel(img_arr, _equalize_channel)
    return clip_uint8(res)


# =====================================================================
# 4. SMOOTHING - SPATIAL DOMAIN (Lowpass & Median)
# =====================================================================
def apply_lowpass_spatial(img_arr: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Neighborhood averaging filter (3x3 atau 5x5): (1 / d) * sum(f)."""
    ksize = int(kernel_size)
    if ksize not in (3, 5, 7):
        raise ValueError("Ukuran kernel harus ganjil (3, 5, atau 7).")
    kernel = np.ones((ksize, ksize), dtype=np.float64) / (ksize * ksize)
    res = convolve2d(img_arr, kernel, flip_kernel=False)
    return clip_uint8(res)


def _median_single(ch: np.ndarray, ksize: int) -> np.ndarray:
    pad = ksize // 2
    padded = np.pad(ch, ((pad, pad), (pad, pad)), mode="edge")
    windows = sliding_window_view(padded, (ksize, ksize))
    return np.median(windows, axis=(-2, -1))


def apply_median_filter(img_arr: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Median Filtering (3x3 atau 5x5) untuk meredam derau impuls."""
    ksize = int(kernel_size)
    if ksize not in (3, 5, 7):
        raise ValueError("Ukuran kernel harus ganjil (3, 5, atau 7).")
    res = apply_per_channel(img_arr, lambda ch: _median_single(ch, ksize))
    return clip_uint8(res)


# =====================================================================
# 5. FREQUENCY DOMAIN HELPER (ILPF, BLPF, IHPF, BHPF)
# =====================================================================
def _distance_matrix(shape) -> np.ndarray:
    rows, cols = shape
    crow, ccol = rows / 2.0, cols / 2.0
    u = np.arange(rows, dtype=np.float64) - crow
    v = np.arange(cols, dtype=np.float64) - ccol
    vv, uu = np.meshgrid(v, u)
    return np.sqrt(uu ** 2 + vv ** 2)


def _apply_freq_filter_channel(ch: np.ndarray, h_filter: np.ndarray) -> np.ndarray:
    f = np.fft.fft2(ch)
    fshift = np.fft.fftshift(f)
    filtered = fshift * h_filter
    ishift = np.fft.ifftshift(filtered)
    img_back = np.fft.ifft2(ishift)
    return np.real(img_back)


def apply_ilpf(img_arr: np.ndarray, d0: float = 30.0) -> np.ndarray:
    """Ideal Lowpass Filter (ILPF): H(u,v) = 1 jika D(u,v) <= D0, else 0."""
    if float(d0) <= 0:
        raise ValueError("Cutoff D0 harus lebih besar dari 0.")
    arr = ensure_float64(img_arr)
    d = _distance_matrix(arr.shape[:2])
    h_filter = (d <= float(d0)).astype(np.float64)
    res = apply_per_channel(arr, lambda ch: _apply_freq_filter_channel(ch, h_filter))
    return clip_uint8(res)


def apply_blpf(img_arr: np.ndarray, d0: float = 30.0, order: int = 2) -> np.ndarray:
    """Butterworth Lowpass Filter (BLPF): H(u,v) = 1 / (1 + (D(u,v)/D0)^(2n))."""
    if float(d0) <= 0 or int(order) <= 0:
        raise ValueError("Cutoff D0 dan Orde n harus lebih besar dari 0.")
    arr = ensure_float64(img_arr)
    d = _distance_matrix(arr.shape[:2])
    h_filter = 1.0 / (1.0 + (d / float(d0)) ** (2 * int(order)))
    res = apply_per_channel(arr, lambda ch: _apply_freq_filter_channel(ch, h_filter))
    return clip_uint8(res)


def apply_ihpf(img_arr: np.ndarray, d0: float = 30.0) -> np.ndarray:
    """Ideal Highpass Filter (IHPF): H(u,v) = 0 jika D(u,v) <= D0, else 1."""
    if float(d0) <= 0:
        raise ValueError("Cutoff D0 harus lebih besar dari 0.")
    arr = ensure_float64(img_arr)
    d = _distance_matrix(arr.shape[:2])
    h_filter = (d > float(d0)).astype(np.float64)
    res = apply_per_channel(arr, lambda ch: _apply_freq_filter_channel(ch, h_filter))
    return clip_uint8(res)


def apply_bhpf(img_arr: np.ndarray, d0: float = 30.0, order: int = 2) -> np.ndarray:
    """Butterworth Highpass Filter (BHPF): H(u,v) = 1 / (1 + (D0/D(u,v))^(2n))."""
    if float(d0) <= 0 or int(order) <= 0:
        raise ValueError("Cutoff D0 dan Orde n harus lebih besar dari 0.")
    arr = ensure_float64(img_arr)
    d = _distance_matrix(arr.shape[:2])
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(d > 1e-12, float(d0) / d, np.inf)
        h_filter = np.where(d > 1e-12, 1.0 / (1.0 + ratio ** (2 * int(order))), 0.0)
    res = apply_per_channel(arr, lambda ch: _apply_freq_filter_channel(ch, h_filter))
    return clip_uint8(res)


# =====================================================================
# 6. SHARPENING - SPATIAL DOMAIN (Highpass & Highboost)
# =====================================================================
def apply_highpass_spatial(img_arr: np.ndarray, kernel_type: str = "sum1") -> np.ndarray:
    """
    Highpass Filtering Spasial:
    - 'sum0': [[-1,-1,-1],[-1,8,-1],[-1,-1,-1]] (jumlah koefisien = 0)
    - 'sum1': [[0,-1,0],[-1,5,-1],[0,-1,0]] (jumlah koefisien = 1)
    """
    if kernel_type == "sum0":
        kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=np.float64)
    else:
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float64)
    res = convolve2d(img_arr, kernel, flip_kernel=True)
    return clip_uint8(res)


def apply_highboost_spatial(img_arr: np.ndarray, boost_factor: float = 1.5) -> np.ndarray:
    """
    Highboost Filtering:
    High-boost = Original + (A - 1) * Highpass, dengan Highpass = Original - Lowpass (A > 1).
    Mempertahankan tingkat kecerahan rata-rata sambil menajamkan detail tepi.
    """
    a = float(boost_factor)
    if a <= 1.0:
        raise ValueError("Faktor penguatan A harus lebih besar dari 1 (A > 1).")
    arr = ensure_float64(img_arr)
    lowpass_kernel = np.ones((3, 3), dtype=np.float64) / 9.0
    lowpass = convolve2d(arr, lowpass_kernel, flip_kernel=False)
    highpass = arr - lowpass
    res = arr + (a - 1.0) * highpass
    return clip_uint8(res)


# =====================================================================
# 7. GEOMETRICS CORRECTION
# =====================================================================
def apply_geometric_correction(
    img_arr: np.ndarray,
    angle_deg: float = 0.0,
    skew_x: float = 0.0,
    skew_y: float = 0.0,
    scale: float = 1.0,
    tx: float = 0.0,
    ty: float = 0.0,
) -> np.ndarray:
    """
    Koreksi distorsi geometris citra (rotasi, kemiringan/skew, skala, dan translasi)
    menggunakan transformasi afin balik (inverse affine mapping) + interpolasi bilinear.
    """
    arr = ensure_float64(img_arr)
    if float(scale) <= 0:
        raise ValueError("Faktor skala harus lebih besar dari 0.")

    h, w = arr.shape[:2]
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0

    theta = np.deg2rad(float(angle_deg))
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    # Matriks transformasi maju: M = Scale * Rotation * Shear
    rot = np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float64)
    shear = np.array([[1.0, float(skew_x)], [float(skew_y), 1.0]], dtype=np.float64)
    mat = float(scale) * (rot @ shear)

    det = np.linalg.det(mat)
    if abs(det) < 1e-8:
        raise ValueError("Kombinasi parameter koreksi geometris menghasilkan matriks singular.")
    inv_mat = np.linalg.inv(mat)

    yy, xx = np.indices((h, w), dtype=np.float64)
    dx = xx - cx - float(tx)
    dy = yy - cy - float(ty)

    src_x = cx + inv_mat[0, 0] * dx + inv_mat[0, 1] * dy
    src_y = cy + inv_mat[1, 0] * dx + inv_mat[1, 1] * dy

    out = _bilinear_sample(arr, src_y, src_x)
    return clip_uint8(out)
