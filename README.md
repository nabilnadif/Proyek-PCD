# Studio PCD Desktop — Aplikasi Pengolahan Citra Digital

Aplikasi desktop **Pengolahan Citra Digital (PCD)** berbasis Python dengan arsitektur **Object-Oriented Programming (OOP)** dan pola **Model-View-Controller (MVC)** modular. Seluruh algoritma inti diimplementasikan menggunakan matriks `NumPy` murni sesuai rumus dan matriks kernel perkuliahan.

## Fitur Utama

1. **File**: `Open...`, `Save`, `Save As`, `Exit` (mendukung `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tif`).
2. **Basic Ops**:
   - `Negative`
   - `Arithmetic`: Add (+), Subtract (-), Multiply (*), Divide (/) — mode Skalar & 2-Citra (auto-resize).
   - `Boolean`: NOT, AND, OR, XOR (auto-resize).
   - `Geometrics`: Translation, Rotation (bilinear inverse mapping), Zooming, Flipping (Horizontal/Vertikal), Cropping.
   - `Thresholding`, `Convolution` (custom kernel 3×3 + preset), `Fourier Transform` (2D FFT log magnitude spectrum).
   - `Colouring`: Binary, Grayscale, RGB (ekstraksi kanal), HSV, CMY, YUV, YIQ, Pseudo-Coloring.
3. **Enhancement**:
   - `Brightness`, `Contrast` (Scaling & Stretching), `Hist. Equalization`.
   - `Smoothing`: Spatial Domain (Lowpass, Median) & Frequency Domain (ILPF, BLPF).
   - `Sharpening`: Spatial Domain (Highpass, Highboost) & Frequency Domain (IHPF, BHPF).
   - `Geometrics correction` (rotasi, skew, skala, translasi).
4. **Noise (6 Model PDF)**:
   - Gaussian, Rayleigh, Erlang (Gamma), Exponential, Uniform, Impulse (Salt & Pepper).
5. **Edge Detection**:
   - `1'st Differential Gradient`: Sobel, Prewitt, Robert.
   - `2'nd Differential Gradient`: Laplacian (4/8-tetangga), LoG (Mexican Hat & Gaussian+Laplacian), Canny (4 tahap lengkap).
   - `Compass` (8 arah mata angin Kirsch).
6. **Segmentation**:
   - `Global (Otsu) & Adaptive Thresholding`, `Contour Detection`, `Hough Transform` (Line & Circle), `Thinning` (Zhang-Suen Skeletonization).
7. **Compression (Bonus)**:
   - Huffman Coding (Lossless), Run-Length Encoding / RLE (Lossless), Uniform Quantization (Lossy) lengkap dengan laporan **Compression Ratio, BPP, MSE, dan PSNR**.
8. **UI/UX**:
   - Dual-Canvas side-by-side (`Original Image` vs `Processed Image`).
   - Quick Toolbar (`Undo` hingga 30 langkah, `Reset to Original`, `Show Histogram`).
   - Status Bar real-time (nama file, resolusi, mode warna, operasi terakhir).

## Struktur Direktori

```text
Aplikasi-PCD-Desktop/
├── main.py                  # Entry point aplikasi
├── requirements.txt         # Daftar library Python
├── README.md                # Dokumentasi proyek
├── ui/
│   ├── app_window.py        # Window utama, Dual-Canvas, State & Pipeline
│   ├── menu_bar.py          # Hierarki Menu Bar & Cascade
│   └── dialogs.py           # Pop-up input parameter, Histogram, About
├── core/
│   ├── utils.py             # Utilitas matriks, konvolusi 2D, clipping
│   ├── basic_ops.py         # Operasi dasar & transformasi ruang warna
│   ├── enhancement.py       # Perbaikan citra spasial & domain frekuensi
│   ├── noise.py             # Generator 6 model PDF noise
│   ├── edge_detection.py    # Deteksi tepi orde-1, orde-2, & compass
│   ├── segmentation.py      # Otsu, Adaptive, Contour, Hough, Thinning
│   └── compression.py       # Huffman, RLE, Kuantisasi, Rasio & PSNR
├── assets/
│   └── sample_images/       # Citra uji siap pakai (RGB, Gray, Binary, dll.)
```

## Cara Instalasi & Menjalankan Aplikasi

```bash
# 1. Clone repositori
git clone https://github.com/nabilnadif/Proyek-PCD
cd Aplikasi-PCD-Desktop

# 2. Buat & aktifkan virtual environment (Sesuaikan dengan versi py anda)
py -3.11 -m venv .venv
# Windows:
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
# macOS / Linux:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Jalankan aplikasi
python main.py

# 5. (Opsional) Jalankan pengujian otomatis seluruh algoritma
python tests/test_all_modules.py
```

## Tim Pengembang

- **Anggota 1 (Lead / UI & Basic Ops)**: Muhammad Nabil Nadif (2407112714)
- **Anggota 2 (Enhancement & Noise)**: Rifaul Khairunisa (2407110731)
- **Anggota 3 (Edge Detection & Segmentation)**: Raka Nadwa (2407113886)

**Mata Kuliah**: Pengolahan Citra Digital (PCD)  
**Dosen Pengampu**: Nanda Dwi Putra, S.Kom., M.Kom.  

---
*Copyright Tim PCD TI-UNRI (2026)*
