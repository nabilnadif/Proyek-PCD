"""
Modul core/compression.py (Fitur Bonus/Opsional sesuai Blueprint)
Implementasi kompresi citra digital beserta perhitungan Rasio Kompresi & PSNR:
- Huffman Coding (Lossless)
- Run-Length Encoding / RLE (Lossless)
- Kuantisasi Seragam / Uniform Quantization (Lossy)
"""
import heapq
import numpy as np
from .utils import ensure_float64, clip_uint8


def compute_mse_psnr(original: np.ndarray, reconstructed: np.ndarray):
    """Menghitung Mean Squared Error (MSE) dan Peak Signal-to-Noise Ratio (PSNR dB)."""
    a = ensure_float64(original)
    b = ensure_float64(reconstructed)
    mse = float(np.mean((a - b) ** 2))
    if mse <= 1e-12:
        return 0.0, float("inf")
    psnr = float(10.0 * np.log10((255.0 ** 2) / mse))
    return mse, psnr


def apply_huffman_compression(img_arr: np.ndarray):
    """
    Simulasi kompresi Lossless Huffman Coding pada seluruh nilai piksel citra.
    Mengembalikan (citra_rekonstruksi, dict_metrik).
    """
    u8 = clip_uint8(ensure_float64(img_arr))
    flat = u8.ravel()
    orig_bits = int(flat.size * 8)

    values, counts = np.unique(flat, return_counts=True)
    if len(values) == 1:
        comp_bits = int(flat.size)
        avg_code_len = 1.0
    else:
        heap = [[int(cnt), [int(val), ""]] for val, cnt in zip(values, counts)]
        heapq.heapify(heap)
        while len(heap) > 1:
            lo = heapq.heappop(heap)
            hi = heapq.heappop(heap)
            for pair in lo[1:]:
                pair[1] = "0" + pair[1]
            for pair in hi[1:]:
                pair[1] = "1" + pair[1]
            heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])

        codebook = {val: code for val, code in heap[0][1:]}
        comp_bits = sum(int(cnt) * len(codebook[int(val)]) for val, cnt in zip(values, counts))
        avg_code_len = comp_bits / float(flat.size)

    cr = orig_bits / float(max(1, comp_bits))
    saving_pct = (1.0 - (comp_bits / float(max(1, orig_bits)))) * 100.0
    mse, psnr = compute_mse_psnr(u8, u8)

    metrics = {
        "method": "Huffman Coding (Lossless)",
        "original_bits": orig_bits,
        "compressed_bits": comp_bits,
        "compression_ratio": cr,
        "bpp": avg_code_len,
        "space_saving_pct": saving_pct,
        "mse": mse,
        "psnr": psnr,
    }
    return u8.copy(), metrics


def apply_rle_compression(img_arr: np.ndarray):
    """
    Simulasi kompresi Lossless Run-Length Encoding (RLE).
    Setiap run disimpan sebagai pasangan (nilai 8-bit, panjang run 8-bit, maks 255).
    Mengembalikan (citra_rekonstruksi, dict_metrik).
    """
    u8 = clip_uint8(ensure_float64(img_arr))
    flat = u8.ravel()
    orig_bits = int(flat.size * 8)

    if flat.size == 0:
        num_runs = 0
    else:
        # Hitung posisi perubahan nilai piksel secara vectorized
        change_idx = np.nonzero(flat[1:] != flat[:-1])[0] + 1
        starts = np.concatenate(([0], change_idx))
        ends = np.concatenate((change_idx, [flat.size]))
        run_lengths = ends - starts
        # Jika panjang run > 255, pecah menjadi beberapa segmen maks 255
        num_runs = int(np.sum((run_lengths + 254) // 255))

    comp_bits = num_runs * 16  # 8-bit value + 8-bit count
    cr = orig_bits / float(max(1, comp_bits))
    bpp = comp_bits / float(max(1, flat.size))
    saving_pct = (1.0 - (comp_bits / float(max(1, orig_bits)))) * 100.0
    mse, psnr = compute_mse_psnr(u8, u8)

    metrics = {
        "method": "Run-Length Encoding / RLE (Lossless)",
        "original_bits": orig_bits,
        "compressed_bits": comp_bits,
        "num_runs": num_runs,
        "compression_ratio": cr,
        "bpp": bpp,
        "space_saving_pct": saving_pct,
        "mse": mse,
        "psnr": psnr,
    }
    return u8.copy(), metrics


def apply_quantization_compression(img_arr: np.ndarray, bits: int = 4):
    """
    Kompresi Lossy menggunakan Kuantisasi Seragam (Uniform Quantization) dari 8-bit ke k-bit (1..7).
    Mengembalikan (citra_terkuantisasi, dict_metrik).
    """
    b = int(bits)
    if b < 1 or b > 7:
        raise ValueError("Jumlah bit kuantisasi harus antara 1 hingga 7.")

    orig = clip_uint8(ensure_float64(img_arr))
    levels = 2 ** b
    step = 256.0 / levels
    quantized = np.floor(orig.astype(np.float64) / step) * step + (step / 2.0)
    recon = clip_uint8(quantized)

    orig_bits = int(orig.size * 8)
    comp_bits = int(orig.size * b)
    cr = 8.0 / float(b)
    saving_pct = (1.0 - (b / 8.0)) * 100.0
    mse, psnr = compute_mse_psnr(orig, recon)

    metrics = {
        "method": f"Uniform Quantization ({b}-bit / {levels} level)",
        "original_bits": orig_bits,
        "compressed_bits": comp_bits,
        "compression_ratio": cr,
        "bpp": float(b),
        "space_saving_pct": saving_pct,
        "mse": mse,
        "psnr": psnr,
    }
    return recon, metrics
