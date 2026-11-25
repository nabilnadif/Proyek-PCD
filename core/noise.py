"""
Modul core/noise.py
Pembangkit 6 model PDF derau (Noise) sesuai Slide M7:
Gaussian, Rayleigh, Erlang (Gamma), Exponential, Uniform, dan Impulse (Salt & Pepper).
"""
import numpy as np
from .utils import ensure_float64, clip_uint8


def apply_gaussian_noise(
    img_arr: np.ndarray, mean: float = 0.0, std: float = 20.0, seed: int = None
) -> np.ndarray:
    """Menambahkan derau Gaussian N(mu, sigma^2)."""
    if float(std) < 0:
        raise ValueError("Standar deviasi (sigma) tidak boleh negatif.")
    rng = np.random.default_rng(seed)
    arr = ensure_float64(img_arr)
    noise = rng.normal(loc=float(mean), scale=float(std), size=arr.shape)
    return clip_uint8(arr + noise)


def apply_rayleigh_noise(
    img_arr: np.ndarray, a: float = 0.0, b: float = 400.0, seed: int = None
) -> np.ndarray:
    """
    Menambahkan derau Rayleigh dengan PDF:
    p(z) = (2/b) * (z - a) * exp(-(z - a)^2 / b) untuk z >= a.
    Inverse CDF: z = a + sqrt(-b * ln(1 - U)).
    """
    if float(b) <= 0:
        raise ValueError("Parameter b pada Rayleigh Noise harus lebih besar dari 0.")
    rng = np.random.default_rng(seed)
    arr = ensure_float64(img_arr)
    u = rng.uniform(1e-12, 1.0 - 1e-12, size=arr.shape)
    noise = float(a) + np.sqrt(-float(b) * np.log(1.0 - u))
    return clip_uint8(arr + noise)


def apply_erlang_noise(
    img_arr: np.ndarray, a: float = 0.05, b: int = 2, seed: int = None
) -> np.ndarray:
    """
    Menambahkan derau Erlang (Gamma) dengan parameter a > 0 dan integer b > 0.
    Distribusi Gamma dengan shape = b dan scale = 1/a.
    """
    if float(a) <= 0 or int(b) <= 0:
        raise ValueError("Parameter a harus > 0 dan integer b harus > 0.")
    rng = np.random.default_rng(seed)
    arr = ensure_float64(img_arr)
    noise = rng.gamma(shape=int(b), scale=1.0 / float(a), size=arr.shape)
    return clip_uint8(arr + noise)


def apply_exponential_noise(
    img_arr: np.ndarray, a: float = 0.05, seed: int = None
) -> np.ndarray:
    """
    Menambahkan derau Exponential (kasus khusus Erlang dengan b = 1, mu = 1/a).
    """
    if float(a) <= 0:
        raise ValueError("Parameter a harus lebih besar dari 0.")
    rng = np.random.default_rng(seed)
    arr = ensure_float64(img_arr)
    noise = rng.exponential(scale=1.0 / float(a), size=arr.shape)
    return clip_uint8(arr + noise)


def apply_uniform_noise(
    img_arr: np.ndarray, low: float = -30.0, high: float = 30.0, seed: int = None
) -> np.ndarray:
    """Menambahkan derau Uniform pada rentang [a, b]."""
    if float(high) < float(low):
        raise ValueError("Batas atas b harus lebih besar atau sama dengan batas bawah a.")
    rng = np.random.default_rng(seed)
    arr = ensure_float64(img_arr)
    noise = rng.uniform(low=float(low), high=float(high), size=arr.shape)
    return clip_uint8(arr + noise)


def apply_impulse_noise(
    img_arr: np.ndarray, p_pepper: float = 0.02, p_salt: float = 0.02, seed: int = None
) -> np.ndarray:
    """
    Menambahkan derau Impulse (Salt & Pepper):
    - Probabilitas Pa (pepper): piksel diubah menjadi 0 (hitam).
    - Probabilitas Pb (salt): piksel diubah menjadi 255 (putih).
    """
    pa = float(p_pepper)
    pb = float(p_salt)
    if pa < 0 or pb < 0 or (pa + pb) > 1.0:
        raise ValueError("Probabilitas Pa dan Pb harus >= 0 dengan Pa + Pb <= 1.")
    rng = np.random.default_rng(seed)
    out = clip_uint8(ensure_float64(img_arr)).copy()

    # Mask spasial (H, W) agar salt/pepper konsisten di semua kanal RGB
    probs = rng.random(out.shape[:2])
    pepper_mask = probs < pa
    salt_mask = (probs >= pa) & (probs < (pa + pb))

    out[pepper_mask] = 0
    out[salt_mask] = 255
    return out
