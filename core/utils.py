"""Utilitas pemrosesan matriks citra murni berbasis NumPy."""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def ensure_float64(img_arr: np.ndarray) -> np.ndarray:
    """Konversi array citra ke float64 untuk menghindari integer overflow."""
    arr = np.asarray(img_arr, dtype=np.float64)
    if arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3]
    return arr


def clip_uint8(arr: np.ndarray) -> np.ndarray:
    """Normalisasi clipping [0, 255] dan konversi ke uint8."""
    return np.clip(np.rint(arr), 0, 255).astype(np.uint8)


def to_grayscale_float(img_arr: np.ndarray) -> np.ndarray:
    """Konversi citra ke grayscale float64 menggunakan rata-rata (R + G + B) / 3."""
    arr = ensure_float64(img_arr)
    if arr.ndim == 2:
        return arr
    return (arr[:, :, 0] + arr[:, :, 1] + arr[:, :, 2]) / 3.0


def match_size_and_channels(arr_a: np.ndarray, arr_b: np.ndarray):
    """Menyamakan dimensi (H, W) dan jumlah kanal citra B agar cocok dengan citra A."""
    from PIL import Image

    a = ensure_float64(arr_a)
    b = ensure_float64(arr_b)

    h_a, w_a = a.shape[:2]
    if b.shape[:2] != (h_a, w_a):
        b_uint8 = clip_uint8(b)
        b_pil = Image.fromarray(b_uint8).resize((w_a, h_a), Image.Resampling.BILINEAR)
        b = np.asarray(b_pil, dtype=np.float64)

    # Samakan jumlah kanal
    if a.ndim == 2 and b.ndim == 3:
        b = to_grayscale_float(b)
    elif a.ndim == 3 and b.ndim == 2:
        b = np.stack([b, b, b], axis=-1)

    return a, b


def convolve2d_single(channel: np.ndarray, kernel: np.ndarray, flip_kernel: bool = True) -> np.ndarray:
    """
    Konvolusi 2D pada satu kanal citra menggunakan sliding_window_view (NumPy vectorized).
    Jika flip_kernel=True, kernel diputar 180 derajat sesuai definisi konvolusi matematis.
    """
    k = np.asarray(kernel, dtype=np.float64)
    if flip_kernel:
        k = np.rot90(k, 2)

    kh, kw = k.shape
    pad_top = kh // 2
    pad_bottom = kh - 1 - pad_top
    pad_left = kw // 2
    pad_right = kw - 1 - pad_left

    padded = np.pad(channel, ((pad_top, pad_bottom), (pad_left, pad_right)), mode="edge")
    windows = sliding_window_view(padded, (kh, kw))
    return np.einsum("ijkl,kl->ij", windows, k, optimize=True)


def convolve2d(img_arr: np.ndarray, kernel: np.ndarray, flip_kernel: bool = True) -> np.ndarray:
    """Konvolusi 2D yang mendukung citra grayscale (2D) maupun RGB (3D)."""
    arr = ensure_float64(img_arr)
    if arr.ndim == 2:
        return convolve2d_single(arr, kernel, flip_kernel=flip_kernel)

    out = np.empty_like(arr)
    for c in range(arr.shape[2]):
        out[:, :, c] = convolve2d_single(arr[:, :, c], kernel, flip_kernel=flip_kernel)
    return out


def apply_per_channel(img_arr: np.ndarray, func) -> np.ndarray:
    """Menjalankan fungsi 2D pada setiap kanal citra (grayscale atau RGB)."""
    arr = ensure_float64(img_arr)
    if arr.ndim == 2:
        return func(arr)
    out = np.empty_like(arr)
    for c in range(arr.shape[2]):
        out[:, :, c] = func(arr[:, :, c])
    return out
