"""
Modul core/basic_ops.py
Implementasi operasi dasar citra digital:
Negative, Arithmetic, Boolean, Geometrics, Thresholding, Convolution,
Fourier Transform, dan Colouring (Binary, Grayscale, RGB, HSV, CMY, YUV, YIQ, Pseudo).
"""
import numpy as np
from .utils import (
    ensure_float64,
    clip_uint8,
    to_grayscale_float,
    match_size_and_channels,
    convolve2d,
)


# =====================================================================
# 1. NEGATIVE
# =====================================================================
def apply_negative(img_arr: np.ndarray) -> np.ndarray:
    """Negatif citra: B(x, y) = 255 - A(x, y)."""
    arr = ensure_float64(img_arr)
    return clip_uint8(255.0 - arr)


# =====================================================================
# 2. ARITHMETIC (+, -, *, /)
# =====================================================================
def apply_arithmetic(
    img_arr: np.ndarray,
    operation: str,
    mode: str = "scalar",
    scalar: float = 0.0,
    second_img: np.ndarray = None,
) -> np.ndarray:
    """
    Operasi aritmatika citra dengan skalar atau citra kedua.
    operation: 'add', 'subtract', 'multiply', 'divide'
    mode: 'scalar' atau 'image'
    """
    a = ensure_float64(img_arr)

    if mode == "image":
        if second_img is None:
            raise ValueError("Citra kedua diperlukan untuk mode 2-citra.")
        a, b = match_size_and_channels(a, second_img)
    else:
        b = float(scalar)

    op = operation.lower()
    if op == "add":
        res = a + b
    elif op in ("subtract", "substract"):
        res = a - b
    elif op == "multiply":
        if mode == "image":
            # Normalisasi agar tidak langsung jenuh di 255 saat perkalian 2 citra
            res = (a * b) / 255.0
        else:
            res = a * b
    elif op == "divide":
        if mode == "scalar":
            if abs(b) < 1e-12:
                raise ZeroDivisionError("Pembagi skalar tidak boleh bernilai 0.")
            res = a / b
        else:
            # Cegah pembagian dengan nol per piksel
            safe_b = np.where(b < 1e-6, 1.0, b)
            res = (a / safe_b) * 255.0
    else:
        raise ValueError(f"Operasi aritmatika tidak dikenal: {operation}")

    return clip_uint8(res)


# =====================================================================
# 3. BOOLEAN (NOT, AND, OR, XOR)
# =====================================================================
def apply_boolean(
    img_arr: np.ndarray,
    operation: str,
    second_img: np.ndarray = None,
    use_binary: bool = False,
    threshold: int = 128,
) -> np.ndarray:
    """
    Operasi logika bitwise/biner: NOT, AND, OR, XOR.
    Jika use_binary=True, citra dikonversi ke biner (0/255) terlebih dahulu.
    """
    op = operation.upper()
    a_u8 = clip_uint8(ensure_float64(img_arr))
    if use_binary:
        a_u8 = np.where(a_u8 >= threshold, 255, 0).astype(np.uint8)

    if op == "NOT":
        return np.bitwise_not(a_u8)

    if second_img is None:
        raise ValueError(f"Citra kedua diperlukan untuk operasi Boolean {op}.")

    a_f, b_f = match_size_and_channels(a_u8, second_img)
    a_u8 = clip_uint8(a_f)
    b_u8 = clip_uint8(b_f)

    if use_binary:
        b_u8 = np.where(b_u8 >= threshold, 255, 0).astype(np.uint8)

    if op == "AND":
        return np.bitwise_and(a_u8, b_u8)
    elif op == "OR":
        return np.bitwise_or(a_u8, b_u8)
    elif op == "XOR":
        return np.bitwise_xor(a_u8, b_u8)
    else:
        raise ValueError(f"Operasi Boolean tidak dikenal: {operation}")


# =====================================================================
# 4. GEOMETRICS (Translation, Rotation, Zooming, Flipping, Cropping)
# =====================================================================
def _bilinear_sample(arr: np.ndarray, src_y: np.ndarray, src_x: np.ndarray) -> np.ndarray:
    """Interpolasi bilinear vectorized untuk pemetaan balik (inverse mapping)."""
    h, w = arr.shape[:2]
    valid = (src_y >= 0) & (src_y <= h - 1) & (src_x >= 0) & (src_x <= w - 1)

    y0 = np.floor(np.clip(src_y, 0, h - 1)).astype(np.int64)
    x0 = np.floor(np.clip(src_x, 0, w - 1)).astype(np.int64)
    y1 = np.minimum(y0 + 1, h - 1)
    x1 = np.minimum(x0 + 1, w - 1)

    dy = np.clip(src_y, 0, h - 1) - y0
    dx = np.clip(src_x, 0, w - 1) - x0

    if arr.ndim == 3:
        dy = dy[:, :, None]
        dx = dx[:, :, None]
        valid = valid[:, :, None]

    top = (1.0 - dx) * arr[y0, x0] + dx * arr[y0, x1]
    bot = (1.0 - dx) * arr[y1, x0] + dx * arr[y1, x1]
    out = (1.0 - dy) * top + dy * bot
    return np.where(valid, out, 0.0)


def apply_translation(img_arr: np.ndarray, shift_row: int, shift_col: int) -> np.ndarray:
    """
    Translasi citra: x' = x + m (baris/vertikal), y' = y + n (kolom/horizontal).
    """
    arr = ensure_float64(img_arr)
    h, w = arr.shape[:2]
    m = int(shift_row)
    n = int(shift_col)

    out = np.zeros_like(arr)
    src_y_start = max(0, -m)
    src_y_end = min(h, h - m)
    src_x_start = max(0, -n)
    src_x_end = min(w, w - n)

    dst_y_start = max(0, m)
    dst_y_end = dst_y_start + (src_y_end - src_y_start)
    dst_x_start = max(0, n)
    dst_x_end = dst_x_start + (src_x_end - src_x_start)

    if src_y_end > src_y_start and src_x_end > src_x_start:
        out[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = arr[
            src_y_start:src_y_end, src_x_start:src_x_end
        ]
    return clip_uint8(out)


def apply_rotation(img_arr: np.ndarray, angle_deg: float) -> np.ndarray:
    """
    Rotasi citra terhadap titik pusat (Xp, Yp) sebesar sudut theta (derajat):
    x' = Xp + (x - Xp)*cos(theta) - (y - Yp)*sin(theta)
    y' = Yp + (x - Xp)*sin(theta) + (y - Yp)*cos(theta)
    Menggunakan inverse mapping + interpolasi bilinear agar hasil mulus tanpa lubang.
    """
    arr = ensure_float64(img_arr)
    h, w = arr.shape[:2]
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0

    theta = np.deg2rad(float(angle_deg))
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    yy, xx = np.indices((h, w), dtype=np.float64)
    dx = xx - cx
    dy = yy - cy

    # Inverse rotation mapping
    src_x = cx + dx * cos_t + dy * sin_t
    src_y = cy - dx * sin_t + dy * cos_t

    out = _bilinear_sample(arr, src_y, src_x)
    return clip_uint8(out)


def apply_zooming(img_arr: np.ndarray, sh_x: float, sh_y: float) -> np.ndarray:
    """
    Penskalaan / Zooming: x' = ShX * x, y' = ShY * y.
    > 1: Zoom In, 0 < skala < 1: Zoom Out.
    """
    arr = ensure_float64(img_arr)
    sx = float(sh_x)
    sy = float(sh_y)
    if sx <= 0 or sy <= 0:
        raise ValueError("Faktor skala ShX dan ShY harus lebih besar dari 0.")

    h, w = arr.shape[:2]
    new_w = max(1, int(round(w * sx)))
    new_h = max(1, int(round(h * sy)))

    yy, xx = np.indices((new_h, new_w), dtype=np.float64)
    src_x = xx / sx
    src_y = yy / sy

    out = _bilinear_sample(arr, src_y, src_x)
    return clip_uint8(out)


def apply_flipping(img_arr: np.ndarray, direction: str = "horizontal") -> np.ndarray:
    """
    Pencerminan citra:
    - Horizontal: y' = 2c - y (flip kiri-kanan)
    - Vertikal: x' = 2c - x (flip atas-bawah)
    """
    arr = clip_uint8(ensure_float64(img_arr))
    d = direction.lower()
    if d in ("horizontal", "h"):
        return np.ascontiguousarray(arr[:, ::-1])
    elif d in ("vertical", "vertikal", "v"):
        return np.ascontiguousarray(arr[::-1, :])
    else:
        raise ValueError("Arah flipping harus 'horizontal' atau 'vertikal'.")


def apply_cropping(img_arr: np.ndarray, x_left: int, x_right: int, y_top: int, y_bottom: int) -> np.ndarray:
    """
    Pemotongan citra pada koordinat (xL, xR, yT, yB).
    Lebar hasil w' = xR - xL, tinggi hasil h' = yB - yT.
    """
    arr = clip_uint8(ensure_float64(img_arr))
    h, w = arr.shape[:2]

    xl = max(0, min(int(x_left), w - 1))
    xr = max(xl + 1, min(int(x_right), w))
    yt = max(0, min(int(y_top), h - 1))
    yb = max(yt + 1, min(int(y_bottom), h))

    return np.ascontiguousarray(arr[yt:yb, xl:xr])


# =====================================================================
# 5. THRESHOLDING
# =====================================================================
def apply_thresholding(img_arr: np.ndarray, threshold: int = 128) -> np.ndarray:
    """
    Pengambangan (Thresholding): f'(x,y) = 0 jika f < T, dan 255 jika f >= T.
    """
    gray = to_grayscale_float(img_arr)
    t = float(threshold)
    out = np.where(gray >= t, 255.0, 0.0)
    return clip_uint8(out)


# =====================================================================
# 6. CONVOLUTION
# =====================================================================
def apply_convolution(img_arr: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Konvolusi 2D dengan matriks kernel (umumnya 3x3).
    Kernel diputar 180 derajat sebelum perkalian korelasi, lalu clipping [0, 255].
    """
    k = np.asarray(kernel, dtype=np.float64)
    if k.ndim != 2:
        raise ValueError("Kernel konvolusi harus berupa matriks 2D.")
    res = convolve2d(img_arr, k, flip_kernel=True)
    return clip_uint8(res)


# =====================================================================
# 7. FOURIER TRANSFORM
# =====================================================================
def apply_fourier_transform(img_arr: np.ndarray) -> np.ndarray:
    """
    Menghitung 2D FFT (np.fft.fft2), menggeser pusat frekuensi (fftshift),
    dan mengembalikan spektrum magnitudo logaritmik log(1 + |F(u,v)|) ternormalisasi [0, 255].
    """
    gray = to_grayscale_float(img_arr)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = np.log1p(np.abs(fshift))

    m_min, m_max = magnitude.min(), magnitude.max()
    if m_max - m_min > 1e-12:
        norm = (magnitude - m_min) / (m_max - m_min) * 255.0
    else:
        norm = np.zeros_like(magnitude)
    return clip_uint8(norm)


# =====================================================================
# 8. COLOURING (Binary, Grayscale, RGB, HSV, CMY, YUV, YIQ, Pseudo)
# =====================================================================
def apply_binary(img_arr: np.ndarray, threshold: int = 128) -> np.ndarray:
    """Konversi grayscale ke citra biner (0 dan 255) dengan threshold T (default 128)."""
    return apply_thresholding(img_arr, threshold=threshold)


def apply_grayscale(img_arr: np.ndarray) -> np.ndarray:
    """Konversi ke grayscale menggunakan rumus f_o(x,y) = (R + G + B) / 3."""
    return clip_uint8(to_grayscale_float(img_arr))


def _ensure_rgb(img_arr: np.ndarray) -> np.ndarray:
    arr = ensure_float64(img_arr)
    if arr.ndim == 2:
        return np.stack([arr, arr, arr], axis=-1)
    return arr[:, :, :3]


def apply_rgb_channel(img_arr: np.ndarray, channel: str = "R") -> np.ndarray:
    """
    Ekstraksi kanal warna RGB secara terpisah ('R', 'G', 'B') atau 'ALL' (Gabungan).
    """
    rgb = _ensure_rgb(img_arr)
    ch = channel.upper()
    out = np.zeros_like(rgb)
    if ch == "R":
        out[:, :, 0] = rgb[:, :, 0]
    elif ch == "G":
        out[:, :, 1] = rgb[:, :, 1]
    elif ch == "B":
        out[:, :, 2] = rgb[:, :, 2]
    elif ch in ("ALL", "GABUNGAN", "RGB"):
        out = rgb
    else:
        raise ValueError("Channel RGB harus 'R', 'G', 'B', atau 'ALL'.")
    return clip_uint8(out)


def apply_hsv(img_arr: np.ndarray) -> np.ndarray:
    """
    Konversi RGB ke ruang warna HSV:
    Normalisasi RGB/255, hitung Cmax, Cmin, Delta, lalu petakan H, S, V ke [0, 255].
    """
    rgb = _ensure_rgb(img_arr) / 255.0
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]

    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    delta = cmax - cmin

    h = np.zeros_like(cmax)
    mask = delta > 1e-12

    r_mask = mask & (cmax == r)
    g_mask = mask & (cmax == g)
    b_mask = mask & (cmax == b)

    h[r_mask] = (60.0 * (((g[r_mask] - b[r_mask]) / delta[r_mask]) % 6.0))
    h[g_mask] = (60.0 * (((b[g_mask] - r[g_mask]) / delta[g_mask]) + 2.0))
    h[b_mask] = (60.0 * (((r[b_mask] - g[b_mask]) / delta[b_mask]) + 4.0))

    s = np.where(cmax > 1e-12, delta / np.where(cmax > 1e-12, cmax, 1.0), 0.0)
    v = cmax

    # Petakan ke [0, 255] untuk visualisasi 3-kanal
    h_u8 = (h / 360.0) * 255.0
    s_u8 = s * 255.0
    v_u8 = v * 255.0

    hsv = np.stack([h_u8, s_u8, v_u8], axis=-1)
    return clip_uint8(hsv)


def apply_cmy(img_arr: np.ndarray) -> np.ndarray:
    """Konversi RGB ke CMY: C = 255 - R, M = 255 - G, Y = 255 - B."""
    rgb = _ensure_rgb(img_arr)
    return clip_uint8(255.0 - rgb)


def apply_yuv(img_arr: np.ndarray) -> np.ndarray:
    """Transformasi matriks linear RGB ke YUV (standar BT.601)."""
    rgb = _ensure_rgb(img_arr)
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]

    y = 0.299 * r + 0.587 * g + 0.114 * b
    u = -0.14713 * r - 0.28886 * g + 0.436 * b + 128.0
    v = 0.615 * r - 0.51499 * g - 0.10001 * b + 128.0

    return clip_uint8(np.stack([y, u, v], axis=-1))


def apply_yiq(img_arr: np.ndarray) -> np.ndarray:
    """Transformasi matriks linear RGB ke YIQ (standar NTSC)."""
    rgb = _ensure_rgb(img_arr)
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]

    y = 0.299 * r + 0.587 * g + 0.114 * b
    i = 0.596 * r - 0.274 * g - 0.322 * b + 128.0
    q = 0.211 * r - 0.523 * g + 0.312 * b + 128.0

    return clip_uint8(np.stack([y, i, q], axis=-1))


def apply_pseudo_color(img_arr: np.ndarray, scheme: str = "jet") -> np.ndarray:
    """
    Pewarnaan semu (Pseudo-coloring / Intensity Slicing) dari intensitas grayscale.
    Pilihan skema: 'jet', 'hot', 'cool', 'rainbow', 'slicing'.
    """
    gray = clip_uint8(to_grayscale_float(img_arr))
    t = gray.astype(np.float64) / 255.0
    sc = scheme.lower()

    if sc == "hot":
        r = np.clip(t * 3.0, 0.0, 1.0)
        g = np.clip(t * 3.0 - 1.0, 0.0, 1.0)
        b = np.clip(t * 3.0 - 2.0, 0.0, 1.0)
        out = np.stack([r, g, b], axis=-1) * 255.0
    elif sc == "cool":
        r = t
        g = 1.0 - t
        b = np.ones_like(t)
        out = np.stack([r, g, b], axis=-1) * 255.0
    elif sc == "rainbow":
        r = np.clip(np.abs(2.0 * t - 0.5) * 2.0 - 0.5, 0.0, 1.0)
        g = np.clip(np.sin(t * np.pi), 0.0, 1.0)
        b = np.clip(np.cos(t * np.pi * 0.5), 0.0, 1.0)
        out = np.stack([r, g, b], axis=-1) * 255.0
    elif sc == "slicing":
        # 8-level intensity slicing klasik
        palette = np.array(
            [
                [0, 0, 128],     # 0-31: Navy
                [0, 0, 255],     # 32-63: Blue
                [0, 192, 255],   # 64-95: Cyan
                [0, 255, 128],   # 96-127: Spring Green
                [128, 255, 0],   # 128-159: Chartreuse
                [255, 255, 0],   # 160-191: Yellow
                [255, 128, 0],   # 192-223: Orange
                [255, 0, 0],     # 224-255: Red
            ],
            dtype=np.float64,
        )
        bins = np.minimum(gray // 32, 7)
        out = palette[bins]
    else:
        # Default: Jet colormap
        r = np.clip(1.5 - np.abs(4.0 * t - 3.0), 0.0, 1.0)
        g = np.clip(1.5 - np.abs(4.0 * t - 2.0), 0.0, 1.0)
        b = np.clip(1.5 - np.abs(4.0 * t - 1.0), 0.0, 1.0)
        out = np.stack([r, g, b], axis=-1) * 255.0

    return clip_uint8(out)
