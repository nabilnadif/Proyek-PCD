"""
Modul ui/menu_bar.py
Konstruksi seluruh hierarki Menu & Cascade persis sesuai blueprint PCD.
"""
import tkinter as tk


def build_menu_bar(root: tk.Tk, app) -> tk.Menu:
    """Membangun Menu Bar lengkap dan mengaitkannya ke method controller di app."""
    menubar = tk.Menu(root)

    # =================================================================
    # 1. MENU FILE
    # =================================================================
    file_menu = tk.Menu(menubar, tearoff=0)
    file_menu.add_command(label="Open...", accelerator="Ctrl+O", command=app.open_image)
    file_menu.add_command(label="Save", accelerator="Ctrl+S", command=app.save_image)
    file_menu.add_command(label="Save As", accelerator="Ctrl+Shift+S", command=app.save_image_as)
    file_menu.add_separator()
    file_menu.add_command(label="Exit", accelerator="Alt+F4", command=app.exit_app)
    menubar.add_cascade(label="File", menu=file_menu)

    # =================================================================
    # 2. MENU BASIC OPS
    # =================================================================
    basic_menu = tk.Menu(menubar, tearoff=0)
    basic_menu.add_command(label="Negative", command=app.op_negative)

    # Arithmetic
    arith_menu = tk.Menu(basic_menu, tearoff=0)
    arith_menu.add_command(label="Add (+)", command=lambda: app.op_arithmetic("add", "Add (+)"))
    arith_menu.add_command(label="Substract (-)", command=lambda: app.op_arithmetic("subtract", "Substract (-)"))
    arith_menu.add_command(label="Multiply (*)", command=lambda: app.op_arithmetic("multiply", "Multiply (*)"))
    arith_menu.add_command(label="Divide (/)", command=lambda: app.op_arithmetic("divide", "Divide (/)"))
    basic_menu.add_cascade(label="Arithmetic", menu=arith_menu)

    # Boolean
    bool_menu = tk.Menu(basic_menu, tearoff=0)
    bool_menu.add_command(label="NOT", command=lambda: app.op_boolean("NOT"))
    bool_menu.add_command(label="AND", command=lambda: app.op_boolean("AND"))
    bool_menu.add_command(label="OR", command=lambda: app.op_boolean("OR"))
    bool_menu.add_command(label="XOR", command=lambda: app.op_boolean("XOR"))
    basic_menu.add_cascade(label="Boolean", menu=bool_menu)

    # Geometrics
    geo_menu = tk.Menu(basic_menu, tearoff=0)
    geo_menu.add_command(label="Translation", command=app.op_translation)
    geo_menu.add_command(label="Rotation", command=app.op_rotation)
    geo_menu.add_command(label="Zooming", command=app.op_zooming)
    geo_menu.add_command(label="Flipping", command=app.op_flipping)
    geo_menu.add_command(label="Cropping", command=app.op_cropping)
    basic_menu.add_cascade(label="Geometrics", menu=geo_menu)

    basic_menu.add_command(label="Thresholding", command=app.op_thresholding)
    basic_menu.add_command(label="Convolution", command=app.op_convolution)
    basic_menu.add_command(label="Fourier Transform", command=app.op_fourier_transform)

    # Colouring
    color_menu = tk.Menu(basic_menu, tearoff=0)
    color_menu.add_command(label="Binary", command=app.op_color_binary)
    color_menu.add_command(label="Grayscale", command=app.op_color_grayscale)
    color_menu.add_command(label="RGB", command=app.op_color_rgb)
    color_menu.add_command(label="HSV", command=app.op_color_hsv)
    color_menu.add_command(label="CMY", command=app.op_color_cmy)
    color_menu.add_command(label="YUV", command=app.op_color_yuv)
    color_menu.add_command(label="YIQ", command=app.op_color_yiq)
    color_menu.add_command(label="Pseudo", command=app.op_color_pseudo)
    basic_menu.add_cascade(label="Colouring", menu=color_menu)

    menubar.add_cascade(label="Basic Ops", menu=basic_menu)

    # =================================================================
    # 3. MENU ENHANCEMENT
    # =================================================================
    enh_menu = tk.Menu(menubar, tearoff=0)
    enh_menu.add_command(label="Brightness", command=app.op_brightness)
    enh_menu.add_command(label="Contrast", command=app.op_contrast)
    enh_menu.add_command(label="Hist. Equalization", command=app.op_hist_equalization)

    # Smoothing
    smooth_menu = tk.Menu(enh_menu, tearoff=0)
    smooth_spatial = tk.Menu(smooth_menu, tearoff=0)
    smooth_spatial.add_command(label="Lowpass Filtering", command=app.op_lowpass_spatial)
    smooth_spatial.add_command(label="Median Filtering", command=app.op_median_spatial)
    smooth_menu.add_cascade(label="Spatial Domain", menu=smooth_spatial)

    smooth_freq = tk.Menu(smooth_menu, tearoff=0)
    smooth_freq.add_command(label="ILPF", command=app.op_ilpf)
    smooth_freq.add_command(label="BLPF", command=app.op_blpf)
    smooth_menu.add_cascade(label="Frequency Domain", menu=smooth_freq)
    enh_menu.add_cascade(label="Smoothing", menu=smooth_menu)

    # Sharpening
    sharp_menu = tk.Menu(enh_menu, tearoff=0)
    sharp_spatial = tk.Menu(sharp_menu, tearoff=0)
    sharp_spatial.add_command(label="Highpass Filtering", command=app.op_highpass_spatial)
    sharp_spatial.add_command(label="Highboost Filtering", command=app.op_highboost_spatial)
    sharp_menu.add_cascade(label="Spatial Domain", menu=sharp_spatial)

    sharp_freq = tk.Menu(sharp_menu, tearoff=0)
    sharp_freq.add_command(label="IHPF", command=app.op_ihpf)
    sharp_freq.add_command(label="BHPF", command=app.op_bhpf)
    sharp_menu.add_cascade(label="Frequency Domain", menu=sharp_freq)
    enh_menu.add_cascade(label="Sharpening", menu=sharp_menu)

    enh_menu.add_command(label="Geometrics correction", command=app.op_geometric_correction)
    menubar.add_cascade(label="Enhancement", menu=enh_menu)

    # =================================================================
    # 4. MENU NOISE
    # =================================================================
    noise_menu = tk.Menu(menubar, tearoff=0)
    noise_menu.add_command(label="Gaussian Noise", command=app.op_noise_gaussian)
    noise_menu.add_command(label="Rayleigh Noise", command=app.op_noise_rayleigh)
    noise_menu.add_command(label="Erlang (Gamma) Noise", command=app.op_noise_erlang)
    noise_menu.add_command(label="Exponential Noise", command=app.op_noise_exponential)
    noise_menu.add_command(label="Uniform Noise", command=app.op_noise_uniform)
    noise_menu.add_command(label="Impulse Noise", command=app.op_noise_impulse)
    menubar.add_cascade(label="Noise", menu=noise_menu)

    # =================================================================
    # 5. MENU EDGE DETECTION
    # =================================================================
    edge_menu = tk.Menu(menubar, tearoff=0)

    first_diff = tk.Menu(edge_menu, tearoff=0)
    first_diff.add_command(label="Sobel", command=app.op_edge_sobel)
    first_diff.add_command(label="Prewitt", command=app.op_edge_prewitt)
    first_diff.add_command(label="Robert", command=app.op_edge_robert)
    edge_menu.add_cascade(label="1'st Differential Gradient", menu=first_diff)

    second_diff = tk.Menu(edge_menu, tearoff=0)
    second_diff.add_command(label="Laplacian", command=app.op_edge_laplacian)
    second_diff.add_command(label="Laplacian of Gaussian (LoG)", command=app.op_edge_log)
    second_diff.add_command(label="Canny", command=app.op_edge_canny)
    edge_menu.add_cascade(label="2'nd Differential Gradient", menu=second_diff)

    edge_menu.add_command(label="Compass", command=app.op_edge_compass)
    menubar.add_cascade(label="Edge Detection", menu=edge_menu)

    # =================================================================
    # 6. MENU SEGMENTATION
    # =================================================================
    seg_menu = tk.Menu(menubar, tearoff=0)

    thresh_seg = tk.Menu(seg_menu, tearoff=0)
    thresh_seg.add_command(label="Global Thresholding (Otsu)", command=app.op_seg_otsu)
    thresh_seg.add_command(label="Adaptive Thresholding", command=app.op_seg_adaptive)
    seg_menu.add_cascade(label="Global & Adaptive Thresholding", menu=thresh_seg)

    seg_menu.add_command(label="Contour Detection", command=app.op_seg_contour)

    hough_menu = tk.Menu(seg_menu, tearoff=0)
    hough_menu.add_command(label="Hough Line", command=app.op_seg_hough_line)
    hough_menu.add_command(label="Hough Circle", command=app.op_seg_hough_circle)
    seg_menu.add_cascade(label="Hough Transform", menu=hough_menu)

    seg_menu.add_command(label="Thinning (Skeletonization)", command=app.op_seg_thinning)
    menubar.add_cascade(label="Segmentation", menu=seg_menu)

    # =================================================================
    # 7. MENU COMPRESSION (Opsional / Bonus Blueprint)
    # =================================================================
    comp_menu = tk.Menu(menubar, tearoff=0)
    comp_menu.add_command(label="Huffman Coding (Lossless)", command=app.op_comp_huffman)
    comp_menu.add_command(label="Run-Length Encoding / RLE (Lossless)", command=app.op_comp_rle)
    comp_menu.add_command(label="Kuantisasi Seragam (Lossy + PSNR)", command=app.op_comp_quantization)
    menubar.add_cascade(label="Compression", menu=comp_menu)

    # =================================================================
    # 8. MENU ABOUT
    # =================================================================
    about_menu = tk.Menu(menubar, tearoff=0)
    about_menu.add_command(label="Info Tim Developer", command=app.show_about)
    about_menu.add_separator()

    tut_menu = tk.Menu(about_menu, tearoff=0)
    tut_menu.add_command(label="Link Github", command=app.open_github_link)
    tut_menu.add_command(label="Link Youtube", command=app.open_youtube_link)
    about_menu.add_cascade(label="Tutorial", menu=tut_menu)

    menubar.add_cascade(label="About", menu=about_menu)

    root.config(menu=menubar)
    return menubar
