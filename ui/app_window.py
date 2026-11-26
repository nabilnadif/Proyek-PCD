"""
Modul ui/app_window.py
Class ImageProcessingApp: Antarmuka Dual-Canvas, Quick Toolbar, Status Bar,
Manajemen State (original_image, current_image, history_stack, file_path),
serta Data Processing Pipeline 6-langkah yang aman dan terintegrasi.
"""
import os
import webbrowser
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
from PIL import Image, ImageTk

from ui.menu_bar import build_menu_bar
from ui.dialogs import (
    MultiFieldDialog,
    SliderDialog,
    ArithmeticDialog,
    ConvolutionDialog,
    ContrastDialog,
    show_histogram_window,
    show_compression_report,
    show_about_dialog,
)
from core import (
    basic_ops,
    enhancement,
    noise,
    edge_detection,
    segmentation,
    compression,
)


# =====================================================================
# KONFIGURASI IDENTITAS APLIKASI & TIM (SILAKAN SESUAIKAN DI SINI)
# =====================================================================
APP_CONFIG = {
    "app_name": "Studio PCD Desktop",
    "team_name": "Tim PCD TI-UNRI",
    "year": "2026",
    "copyright": "Copyright Tim PCD TI-UNRI (2026)",
    "course": "Pengolahan Citra Digital (PCD)",
    "lecturer": "Nanda Dwi Putra, S.Kom., M.Kom.",
    "institution": "Teknik Informatika - Fakultas Teknik Universitas Riau",
    "members": [
        "Muhammad Nabil Nadif (2407112714) - Lead / UI & Basic Ops",
        "Rifaul Khairunisa (2407110731) - Enhancement & Noise",
        "Raka Nadwa (2407113886) - Edge Detection & Segmentation",
    ],
    "github_url": "https://github.com/nabilnadif/Proyek-PCD",
    "youtube_url": "https://www.youtube.com/watch?v=VIDEO_TUTORIAL_ID",
}


class ImageProcessingApp:
    """Controller & View utama Aplikasi Pengolahan Citra Digital."""

    MAX_HISTORY = 30

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title(f"{APP_CONFIG['app_name']} - {APP_CONFIG['copyright']}")
        self.root.geometry("1120x680")
        self.root.minsize(820, 600)

        # Manajemen State Citra (sesuai Blueprint)
        self.original_image: Image.Image = None
        self.current_image: Image.Image = None
        self.history_stack: list = []
        self.file_path: str = None
        self.last_operation: str = "Siap"

        # Referensi PhotoImage agar tidak terhapus oleh garbage collector
        self._tk_orig = None
        self._tk_proc = None
        self._resize_job = None

        self._configure_style()
        build_menu_bar(self.root, self)
        self._build_toolbar()
        self._build_dual_canvas()
        self._build_status_bar()
        self._bind_shortcuts()

    def _configure_style(self):
        style = ttk.Style(self.root)
        for theme in ("vista", "clam", "default"):
            if theme in style.theme_names():
                style.theme_use(theme)
                break

    def _build_toolbar(self):
        toolbar = ttk.Frame(self.root, padding=(8, 5))
        toolbar.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(toolbar, text="📂 Open", command=self.open_image).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(toolbar, text="💾 Save", command=self.save_image).pack(side=tk.LEFT, padx=(0, 8))

        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)

        self.btn_undo = ttk.Button(toolbar, text="↩ Undo (0)", command=self.undo, state=tk.DISABLED)
        self.btn_undo.pack(side=tk.LEFT, padx=(0, 4))

        self.btn_reset = ttk.Button(toolbar, text="⟲ Reset to Original", command=self.reset_to_original, state=tk.DISABLED)
        self.btn_reset.pack(side=tk.LEFT, padx=(0, 4))

        self.btn_hist = ttk.Button(toolbar, text="📊 Show Histogram", command=self.show_histogram, state=tk.DISABLED)
        self.btn_hist.pack(side=tk.LEFT, padx=(0, 4))

        ttk.Label(
            toolbar,
            text="Tips: Gunakan Menu Bar di atas untuk memilih operasi citra.",
            foreground="#555555",
        ).pack(side=tk.RIGHT, padx=6)

    def _build_dual_canvas(self):
        main_pane = ttk.Frame(self.root, padding=(8, 4))
        main_pane.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Panel Kiri: Original Image
        left_frame = ttk.LabelFrame(main_pane, text="Original Image", padding=4)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 4))

        self.canvas_original = tk.Canvas(left_frame, bg="gray", highlightthickness=0)
        self.canvas_original.pack(fill=tk.BOTH, expand=True)

        # Panel Kanan: Processed Image
        right_frame = ttk.LabelFrame(main_pane, text="Processed Image", padding=4)
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(4, 0))

        self.canvas_processed = tk.Canvas(right_frame, bg="gray", highlightthickness=0)
        self.canvas_processed.pack(fill=tk.BOTH, expand=True)

        self.canvas_original.bind("<Configure>", self._on_canvas_resize)
        self.canvas_processed.bind("<Configure>", self._on_canvas_resize)

        self._draw_placeholder_text()

    def _build_status_bar(self):
        status_frame = ttk.Frame(self.root, relief=tk.SUNKEN, padding=(8, 4))
        status_frame.pack(side=tk.BOTTOM, fill=tk.X)

        self.status_file_var = tk.StringVar(value="File: -")
        self.status_res_var = tk.StringVar(value="Resolusi: -")
        self.status_mode_var = tk.StringVar(value="Mode: -")
        self.status_op_var = tk.StringVar(value="Operasi Terakhir: Siap")

        ttk.Label(status_frame, textvariable=self.status_file_var, width=28, anchor=tk.W).pack(side=tk.LEFT)
        ttk.Separator(status_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)
        ttk.Label(status_frame, textvariable=self.status_res_var, width=20, anchor=tk.W).pack(side=tk.LEFT)
        ttk.Separator(status_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)
        ttk.Label(status_frame, textvariable=self.status_mode_var, width=14, anchor=tk.W).pack(side=tk.LEFT)
        ttk.Separator(status_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)
        ttk.Label(status_frame, textvariable=self.status_op_var, anchor=tk.W).pack(side=tk.LEFT, fill=tk.X, expand=True)

    def _bind_shortcuts(self):
        self.root.bind("<Control-o>", lambda _e: self.open_image())
        self.root.bind("<Control-s>", lambda _e: self.save_image())
        self.root.bind("<Control-S>", lambda _e: self.save_image_as())
        self.root.bind("<Control-z>", lambda _e: self.undo())
        self.root.bind("<Control-r>", lambda _e: self.reset_to_original())
        self.root.bind("<Control-h>", lambda _e: self.show_histogram())

    # =================================================================
    # RENDERING & MANAJEMEN STATE
    # =================================================================
    def _draw_placeholder_text(self):
        for canvas, label in (
            (self.canvas_original, "Buka citra melalui File -> Open... (Ctrl+O)"),
            (self.canvas_processed, "Hasil pemrosesan akan tampil di sini"),
        ):
            canvas.delete("all")
            w = max(canvas.winfo_width(), 400)
            h = max(canvas.winfo_height(), 300)
            canvas.create_text(w // 2, h // 2, text=label, fill="#f0f0f0", font=("Segoe UI", 10))

    def _on_canvas_resize(self, _event):
        if self._resize_job is not None:
            self.root.after_cancel(self._resize_job)
        self._resize_job = self.root.after(60, self.display_images)

    def _render_to_canvas(self, canvas: tk.Canvas, pil_img: Image.Image):
        canvas.delete("all")
        cw = max(canvas.winfo_width(), 100)
        ch = max(canvas.winfo_height(), 100)

        img_copy = pil_img.copy()
        img_copy.thumbnail((cw - 16, ch - 16), Image.Resampling.LANCZOS)
        tk_img = ImageTk.PhotoImage(img_copy)
        canvas.create_image(cw // 2, ch // 2, image=tk_img, anchor=tk.CENTER)
        return tk_img

    def display_images(self):
        if self.original_image is None or self.current_image is None:
            self._draw_placeholder_text()
            return
        self._tk_orig = self._render_to_canvas(self.canvas_original, self.original_image)
        self._tk_proc = self._render_to_canvas(self.canvas_processed, self.current_image)

    def update_controls_and_status(self):
        has_img = self.current_image is not None
        self.btn_undo.configure(
            text=f"↩ Undo ({len(self.history_stack)})",
            state=tk.NORMAL if len(self.history_stack) > 0 else tk.DISABLED,
        )
        self.btn_reset.configure(state=tk.NORMAL if has_img else tk.DISABLED)
        self.btn_hist.configure(state=tk.NORMAL if has_img else tk.DISABLED)

        if has_img:
            fname = os.path.basename(self.file_path) if self.file_path else "Untitled"
            w, h = self.current_image.size
            mode = self.current_image.mode
            self.status_file_var.set(f"File: {fname}")
            self.status_res_var.set(f"Resolusi: {w} x {h} px")
            self.status_mode_var.set(f"Mode: {mode}")
        else:
            self.status_file_var.set("File: -")
            self.status_res_var.set("Resolusi: -")
            self.status_mode_var.set("Mode: -")

        self.status_op_var.set(f"Operasi Terakhir: {self.last_operation}")

    # =================================================================
    # DATA PROCESSING PIPELINE (6 LANGKAH SESUAI BLUEPRINT)
    # =================================================================
    def _ensure_image_loaded(self) -> bool:
        if self.current_image is None:
            messagebox.showwarning("Peringatan", "Buka citra terlebih dahulu!")
            return False
        return True

    def _run_pipeline(self, op_name: str, core_func, *args, **kwargs):
        """
        Siklus standar 6 langkah:
        1. Validasi Citra
        2. Konversi PIL.Image -> np.float64
        3. Eksekusi fungsi core/
        4. Clipping [0, 255] uint8
        5. Konversi kembali -> PIL.Image, push history_stack
        6. Render Dual-Canvas & update Status Bar
        """
        if not self._ensure_image_loaded():
            return None

        try:
            self.root.config(cursor="watch")
            self.root.update_idletasks()

            arr = np.asarray(self.current_image, dtype=np.float64)
            output = core_func(arr, *args, **kwargs)

            extra_info = None
            if isinstance(output, tuple):
                res_arr, extra_info = output
            else:
                res_arr = output

            res_u8 = np.clip(np.rint(res_arr), 0, 255).astype(np.uint8)
            new_img = Image.fromarray(res_u8)

            self.history_stack.append(self.current_image.copy())
            if len(self.history_stack) > self.MAX_HISTORY:
                self.history_stack.pop(0)

            self.current_image = new_img
            self.last_operation = op_name
            self.display_images()
            self.update_controls_and_status()
            return extra_info
        except Exception as exc:
            messagebox.showerror("Error Pemrosesan", f"Gagal menjalankan {op_name}:\n{exc}")
            return None
        finally:
            self.root.config(cursor="")

    # =================================================================
    # TOOLBAR ACTIONS (UNDO, RESET, HISTOGRAM)
    # =================================================================
    def undo(self):
        if not self.history_stack:
            return
        self.current_image = self.history_stack.pop()
        self.last_operation = "Undo"
        self.display_images()
        self.update_controls_and_status()

    def reset_to_original(self):
        if not self._ensure_image_loaded():
            return
        self.history_stack.append(self.current_image.copy())
        self.current_image = self.original_image.copy()
        self.last_operation = "Reset to Original"
        self.display_images()
        self.update_controls_and_status()

    def show_histogram(self):
        if not self._ensure_image_loaded():
            return
        show_histogram_window(self.root, self.original_image, self.current_image)

    # =================================================================
    # A. MENU FILE
    # =================================================================
    def open_image(self):
        path = filedialog.askopenfilename(
            title="Pilih Citra Digital",
            filetypes=[
                ("Image Files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"),
                ("All Files", "*.*"),
            ],
        )
        if not path:
            return
        try:
            img = Image.open(path)
            # Normalisasi mode RGBA/P ke RGB agar konsisten di seluruh pipeline
            if img.mode in ("RGBA", "P", "LA"):
                img = img.convert("RGB")
            elif img.mode == "1":
                img = img.convert("L")

            self.original_image = img.copy()
            self.current_image = img.copy()
            self.history_stack.clear()
            self.file_path = path
            self.last_operation = "Open Image"
            self.display_images()
            self.update_controls_and_status()
        except Exception as exc:
            messagebox.showerror("Gagal Membuka Citra", f"File tidak dapat dibuka:\n{exc}")

    def save_image(self):
        if not self._ensure_image_loaded():
            return
        if not self.file_path:
            self.save_image_as()
            return
        try:
            self.current_image.save(self.file_path)
            self.last_operation = "Save"
            self.update_controls_and_status()
            messagebox.showinfo("Berhasil", f"Citra berhasil disimpan ke:\n{self.file_path}")
        except Exception as exc:
            messagebox.showerror("Gagal Menyimpan", str(exc))

    def save_image_as(self):
        if not self._ensure_image_loaded():
            return
        path = filedialog.asksaveasfilename(
            title="Simpan Citra Sebagai",
            defaultextension=".png",
            filetypes=[
                ("PNG Image", "*.png"),
                ("JPEG Image", "*.jpg *.jpeg"),
                ("BMP Image", "*.bmp"),
                ("TIFF Image", "*.tif *.tiff"),
            ],
        )
        if not path:
            return
        try:
            img_to_save = self.current_image
            if path.lower().endswith((".jpg", ".jpeg")) and img_to_save.mode not in ("RGB", "L"):
                img_to_save = img_to_save.convert("RGB")
            img_to_save.save(path)
            self.file_path = path
            self.last_operation = "Save As"
            self.update_controls_and_status()
            messagebox.showinfo("Berhasil", f"Citra berhasil disimpan ke:\n{path}")
        except Exception as exc:
            messagebox.showerror("Gagal Menyimpan", str(exc))

    def exit_app(self):
        self.root.destroy()

    # =================================================================
    # B. MENU BASIC OPS
    # =================================================================
    def op_negative(self):
        self._run_pipeline("Basic Ops - Negative", basic_ops.apply_negative)

    def op_arithmetic(self, op_key: str, op_label: str):
        if not self._ensure_image_loaded():
            return
        dlg = ArithmeticDialog(self.root, op_label)
        if dlg.result is None:
            return
        self._run_pipeline(
            f"Arithmetic - {op_label} ({dlg.result['mode']})",
            basic_ops.apply_arithmetic,
            operation=op_key,
            mode=dlg.result["mode"],
            scalar=dlg.result["scalar"],
            second_img=dlg.result["second_img"],
        )

    def op_boolean(self, op_key: str):
        if not self._ensure_image_loaded():
            return
        if op_key == "NOT":
            self._run_pipeline("Boolean - NOT", basic_ops.apply_boolean, operation="NOT")
            return

        path = filedialog.askopenfilename(
            title=f"Pilih Citra Kedua (B) untuk Boolean {op_key}",
            filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff")],
        )
        if not path:
            return
        second_arr = np.asarray(Image.open(path), dtype=np.float64)
        self._run_pipeline(
            f"Boolean - {op_key}",
            basic_ops.apply_boolean,
            operation=op_key,
            second_img=second_arr,
        )

    def op_translation(self):
        if not self._ensure_image_loaded():
            return
        dlg = MultiFieldDialog(
            self.root,
            "Geometrics - Translation",
            [
                {"name": "m", "label": "Geser Baris (m / vertikal):", "type": "int", "default": 30},
                {"name": "n", "label": "Geser Kolom (n / horizontal):", "type": "int", "default": 40},
            ],
            subtitle="Rumus: x' = x + m, y' = y + n",
        )
        if dlg.result is None:
            return
        self._run_pipeline(
            f"Translation (m={dlg.result['m']}, n={dlg.result['n']})",
            basic_ops.apply_translation,
            shift_row=dlg.result["m"],
            shift_col=dlg.result["n"],
        )

    def op_rotation(self):
        if not self._ensure_image_loaded():
            return
        dlg = SliderDialog(self.root, "Geometrics - Rotation", "Sudut Rotasi (derajat):", -360, 360, 45, is_int=False)
        if dlg.result is None:
            return
        self._run_pipeline(f"Rotation ({dlg.result}°)", basic_ops.apply_rotation, angle_deg=dlg.result)

    def op_zooming(self):
        if not self._ensure_image_loaded():
            return
        dlg = MultiFieldDialog(
            self.root,
            "Geometrics - Zooming",
            [
                {"name": "sx", "label": "Faktor Skala ShX (>0):", "type": "float", "default": 1.5, "min": 0.1, "max": 5.0},
                {"name": "sy", "label": "Faktor Skala ShY (>0):", "type": "float", "default": 1.5, "min": 0.1, "max": 5.0},
            ],
            subtitle="> 1: Zoom In, 0 < skala < 1: Zoom Out",
        )
        if dlg.result is None:
            return
        self._run_pipeline(
            f"Zooming ({dlg.result['sx']}x, {dlg.result['sy']}x)",
            basic_ops.apply_zooming,
            sh_x=dlg.result["sx"],
            sh_y=dlg.result["sy"],
        )

    def op_flipping(self):
        if not self._ensure_image_loaded():
            return
        dlg = MultiFieldDialog(
            self.root,
            "Geometrics - Flipping",
            [{"name": "dir", "label": "Arah Pencerminan:", "type": "choice", "default": "Horizontal", "choices": ["Horizontal", "Vertikal"]}],
        )
        if dlg.result is None:
            return
        self._run_pipeline(
            f"Flipping ({dlg.result['dir']})",
            basic_ops.apply_flipping,
            direction=dlg.result["dir"].lower(),
        )

    def op_cropping(self):
        if not self._ensure_image_loaded():
            return
        w, h = self.current_image.size
        dlg = MultiFieldDialog(
            self.root,
            "Geometrics - Cropping",
            [
                {"name": "xl", "label": f"xL (0 .. {w-1}):", "type": "int", "default": w // 4, "min": 0, "max": w - 1},
                {"name": "xr", "label": f"xR (1 .. {w}):", "type": "int", "default": (3 * w) // 4, "min": 1, "max": w},
                {"name": "yt", "label": f"yT (0 .. {h-1}):", "type": "int", "default": h // 4, "min": 0, "max": h - 1},
                {"name": "yb", "label": f"yB (1 .. {h}):", "type": "int", "default": (3 * h) // 4, "min": 1, "max": h},
            ],
            subtitle=f"Dimensi citra saat ini: {w} x {h} piksel",
        )
        if dlg.result is None:
            return
        self._run_pipeline(
            f"Cropping ({dlg.result['xl']}:{dlg.result['xr']}, {dlg.result['yt']}:{dlg.result['yb']})",
            basic_ops.apply_cropping,
            x_left=dlg.result["xl"],
            x_right=dlg.result["xr"],
            y_top=dlg.result["yt"],
            y_bottom=dlg.result["yb"],
        )

    def op_thresholding(self):
        if not self._ensure_image_loaded():
            return
        dlg = SliderDialog(self.root, "Basic Ops - Thresholding", "Nilai Ambang T (0 - 255):", 0, 255, 128, is_int=True)
        if dlg.result is None:
            return
        self._run_pipeline(f"Thresholding (T={dlg.result})", basic_ops.apply_thresholding, threshold=dlg.result)

    def op_convolution(self):
        if not self._ensure_image_loaded():
            return
        dlg = ConvolutionDialog(self.root)
        if dlg.result is None:
            return
        self._run_pipeline("Convolution 3x3", basic_ops.apply_convolution, kernel=dlg.result)

    def op_fourier_transform(self):
        self._run_pipeline("2D Fourier Transform (Magnitude Spectrum)", basic_ops.apply_fourier_transform)

    def op_color_binary(self):
        if not self._ensure_image_loaded():
            return
        dlg = SliderDialog(self.root, "Colouring - Binary", "Nilai Ambang T (default 128):", 0, 255, 128, is_int=True)
        if dlg.result is None:
            return
        self._run_pipeline(f"Colouring - Binary (T={dlg.result})", basic_ops.apply_binary, threshold=dlg.result)

    def op_color_grayscale(self):
        self._run_pipeline("Colouring - Grayscale", basic_ops.apply_grayscale)

    def op_color_rgb(self):
        if not self._ensure_image_loaded():
            return
        dlg = MultiFieldDialog(
            self.root,
            "Colouring - Ekstraksi Kanal RGB",
            [{"name": "ch", "label": "Pilih Kanal Warna:", "type": "choice", "default": "R", "choices": ["R", "G", "B", "Gabungan"]}],
        )
        if dlg.result is None:
            return
        self._run_pipeline(f"Colouring - RGB ({dlg.result['ch']})", basic_ops.apply_rgb_channel, channel=dlg.result["ch"])

    def op_color_hsv(self):
        self._run_pipeline("Colouring - HSV", basic_ops.apply_hsv)

    def op_color_cmy(self):
        self._run_pipeline("Colouring - CMY", basic_ops.apply_cmy)

    def op_color_yuv(self):
        self._run_pipeline("Colouring - YUV", basic_ops.apply_yuv)

    def op_color_yiq(self):
        self._run_pipeline("Colouring - YIQ", basic_ops.apply_yiq)

    def op_color_pseudo(self):
        if not self._ensure_image_loaded():
            return
        dlg = MultiFieldDialog(
            self.root,
            "Colouring - Pseudo Color",
            [{"name": "scheme", "label": "Skema Pewarnaan (Colormap):", "type": "choice", "default": "jet", "choices": ["jet", "hot", "cool", "rainbow", "slicing"]}],
        )
        if dlg.result is None:
            return
        self._run_pipeline(f"Colouring - Pseudo ({dlg.result['scheme']})", basic_ops.apply_pseudo_color, scheme=dlg.result["scheme"])

    # =================================================================
    # C. MENU ENHANCEMENT
    # =================================================================
    def op_brightness(self):
        if not self._ensure_image_loaded():
            return
        dlg = SliderDialog(self.root, "Enhancement - Brightness", "Konstanta k (-255 s.d. +255):", -255, 255, 35, is_int=True)
        if dlg.result is None:
            return
        self._run_pipeline(f"Brightness (k={dlg.result:+d})", enhancement.apply_brightness, k=dlg.result)

    def op_contrast(self):
        if not self._ensure_image_loaded():
            return
        arr = np.asarray(self.current_image)
        dlg = ContrastDialog(self.root, int(arr.min()), int(arr.max()))
        if dlg.result is None:
            return
        if dlg.result["mode"] == "scaling":
            self._run_pipeline(
                f"Contrast Scaling (G={dlg.result['gain']}, P={dlg.result['pivot']})",
                enhancement.apply_contrast_scaling,
                gain=dlg.result["gain"],
                pivot=dlg.result["pivot"],
            )
        else:
            self._run_pipeline(
                f"Contrast Stretching ({dlg.result['r_min']}-{dlg.result['r_max']})",
                enhancement.apply_contrast_stretching,
                r_min=dlg.result["r_min"],
                r_max=dlg.result["r_max"],
            )

    def op_hist_equalization(self):
        self._run_pipeline("Histogram Equalization", enhancement.apply_histogram_equalization)

    def op_lowpass_spatial(self):
        if not self._ensure_image_loaded():
            return
        dlg = MultiFieldDialog(
            self.root,
            "Smoothing - Lowpass Filtering",
            [{"name": "k", "label": "Ukuran Kernel:", "type": "choice", "default": "3x3", "choices": ["3x3", "5x5", "7x7"]}],
        )
        if dlg.result is None:
            return
        ksize = int(dlg.result["k"][0])
        self._run_pipeline(f"Lowpass Spatial ({ksize}x{ksize})", enhancement.apply_lowpass_spatial, kernel_size=ksize)

    def op_median_spatial(self):
        if not self._ensure_image_loaded():
            return
        dlg = MultiFieldDialog(
            self.root,
            "Smoothing - Median Filtering",
            [{"name": "k", "label": "Ukuran Kernel:", "type": "choice", "default": "3x3", "choices": ["3x3", "5x5"]}],
        )
        if dlg.result is None:
            return
        ksize = int(dlg.result["k"][0])
        self._run_pipeline(f"Median Filter ({ksize}x{ksize})", enhancement.apply_median_filter, kernel_size=ksize)

    def op_ilpf(self):
        if not self._ensure_image_loaded():
            return
        dlg = SliderDialog(self.root, "Smoothing Frequency - ILPF", "Frekuensi Cutoff D0:", 1, 250, 40, is_int=False)
        if dlg.result is None:
            return
        self._run_pipeline(f"ILPF (D0={dlg.result})", enhancement.apply_ilpf, d0=dlg.result)

    def op_blpf(self):
        if not self._ensure_image_loaded():
            return
        dlg = MultiFieldDialog(
            self.root,
            "Smoothing Frequency - BLPF",
            [
                {"name": "d0", "label": "Frekuensi Cutoff D0 (>0):", "type": "float", "default": 40.0, "min": 1.0},
                {"name": "n", "label": "Orde Butterworth n (>=1):", "type": "int", "default": 2, "min": 1, "max": 10},
            ],
        )
        if dlg.result is None:
            return
        self._run_pipeline(
            f"BLPF (D0={dlg.result['d0']}, n={dlg.result['n']})",
            enhancement.apply_blpf,
            d0=dlg.result["d0"],
            order=dlg.result["n"],
        )

    def op_highpass_spatial(self):
        if not self._ensure_image_loaded():
            return
        dlg = MultiFieldDialog(
            self.root,
            "Sharpening - Highpass Filtering",
            [{"name": "kt", "label": "Pilihan Kernel:", "type": "choice", "default": "Sum = 1 (Sharpen)", "choices": ["Sum = 1 (Sharpen)", "Sum = 0 (Edge)"]}],
        )
        if dlg.result is None:
            return
        ktype = "sum0" if "0" in dlg.result["kt"] else "sum1"
        self._run_pipeline(f"Highpass Spatial ({dlg.result['kt']})", enhancement.apply_highpass_spatial, kernel_type=ktype)

    def op_highboost_spatial(self):
        if not self._ensure_image_loaded():
            return
        dlg = SliderDialog(self.root, "Sharpening - Highboost Filtering", "Faktor Penguatan A (A > 1):", 1.1, 5.0, 1.8, is_int=False)
        if dlg.result is None:
            return
        self._run_pipeline(f"Highboost Spatial (A={dlg.result})", enhancement.apply_highboost_spatial, boost_factor=dlg.result)

    def op_ihpf(self):
        if not self._ensure_image_loaded():
            return
        dlg = SliderDialog(self.root, "Sharpening Frequency - IHPF", "Frekuensi Cutoff D0:", 1, 250, 30, is_int=False)
        if dlg.result is None:
            return
        self._run_pipeline(f"IHPF (D0={dlg.result})", enhancement.apply_ihpf, d0=dlg.result)

    def op_bhpf(self):
        if not self._ensure_image_loaded():
            return
        dlg = MultiFieldDialog(
            self.root,
            "Sharpening Frequency - BHPF",
            [
                {"name": "d0", "label": "Frekuensi Cutoff D0 (>0):", "type": "float", "default": 30.0, "min": 1.0},
                {"name": "n", "label": "Orde Butterworth n (>=1):", "type": "int", "default": 2, "min": 1, "max": 10},
            ],
        )
        if dlg.result is None:
            return
        self._run_pipeline(
            f"BHPF (D0={dlg.result['d0']}, n={dlg.result['n']})",
            enhancement.apply_bhpf,
            d0=dlg.result["d0"],
            order=dlg.result["n"],
        )

    def op_geometric_correction(self):
        if not self._ensure_image_loaded():
            return
        dlg = MultiFieldDialog(
            self.root,
            "Enhancement - Geometrics Correction",
            [
                {"name": "angle", "label": "Rotasi Koreksi (derajat):", "type": "float", "default": -5.0},
                {"name": "sx", "label": "Koreksi Skew X:", "type": "float", "default": 0.1},
                {"name": "sy", "label": "Koreksi Skew Y:", "type": "float", "default": 0.0},
                {"name": "scale", "label": "Skala (>0):", "type": "float", "default": 1.0, "min": 0.1, "max": 4.0},
                {"name": "tx", "label": "Translasi X (px):", "type": "float", "default": 0.0},
                {"name": "ty", "label": "Translasi Y (px):", "type": "float", "default": 0.0},
            ],
        )
        if dlg.result is None:
            return
        self._run_pipeline(
            "Geometrics Correction",
            enhancement.apply_geometric_correction,
            angle_deg=dlg.result["angle"],
            skew_x=dlg.result["sx"],
            skew_y=dlg.result["sy"],
            scale=dlg.result["scale"],
            tx=dlg.result["tx"],
            ty=dlg.result["ty"],
        )

    # =================================================================
    # D. MENU NOISE
    # =================================================================
    def op_noise_gaussian(self):
        if not self._ensure_image_loaded():
            return
        dlg = MultiFieldDialog(
            self.root,
            "Gaussian Noise",
            [
                {"name": "mean", "label": "Mean (μ):", "type": "float", "default": 0.0},
                {"name": "std", "label": "Std Dev (σ >= 0):", "type": "float", "default": 25.0, "min": 0.0},
            ],
        )
        if dlg.result is None:
            return
        self._run_pipeline(
            f"Gaussian Noise (μ={dlg.result['mean']}, σ={dlg.result['std']})",
            noise.apply_gaussian_noise,
            mean=dlg.result["mean"],
            std=dlg.result["std"],
        )

    def op_noise_rayleigh(self):
        if not self._ensure_image_loaded():
            return
        dlg = MultiFieldDialog(
            self.root,
            "Rayleigh Noise",
            [
                {"name": "a", "label": "Parameter a:", "type": "float", "default": 0.0},
                {"name": "b", "label": "Parameter b (>0):", "type": "float", "default": 500.0, "min": 0.01},
            ],
        )
        if dlg.result is None:
            return
        self._run_pipeline(
            f"Rayleigh Noise (a={dlg.result['a']}, b={dlg.result['b']})",
            noise.apply_rayleigh_noise,
            a=dlg.result["a"],
            b=dlg.result["b"],
        )

    def op_noise_erlang(self):
        if not self._ensure_image_loaded():
            return
        dlg = MultiFieldDialog(
            self.root,
            "Erlang (Gamma) Noise",
            [
                {"name": "a", "label": "Parameter a (>0):", "type": "float", "default": 0.05, "min": 0.001},
                {"name": "b", "label": "Integer b (>0):", "type": "int", "default": 2, "min": 1},
            ],
        )
        if dlg.result is None:
            return
        self._run_pipeline(
            f"Erlang Noise (a={dlg.result['a']}, b={dlg.result['b']})",
            noise.apply_erlang_noise,
            a=dlg.result["a"],
            b=dlg.result["b"],
        )

    def op_noise_exponential(self):
        if not self._ensure_image_loaded():
            return
        dlg = MultiFieldDialog(
            self.root,
            "Exponential Noise",
            [{"name": "a", "label": "Parameter a (>0):", "type": "float", "default": 0.04, "min": 0.001}],
        )
        if dlg.result is None:
            return
        self._run_pipeline(
            f"Exponential Noise (a={dlg.result['a']})",
            noise.apply_exponential_noise,
            a=dlg.result["a"],
        )

    def op_noise_uniform(self):
        if not self._ensure_image_loaded():
            return
        dlg = MultiFieldDialog(
            self.root,
            "Uniform Noise",
            [
                {"name": "low", "label": "Batas Bawah (a):", "type": "float", "default": -35.0},
                {"name": "high", "label": "Batas Atas (b):", "type": "float", "default": 35.0},
            ],
        )
        if dlg.result is None:
            return
        self._run_pipeline(
            f"Uniform Noise [{dlg.result['low']}, {dlg.result['high']}]",
            noise.apply_uniform_noise,
            low=dlg.result["low"],
            high=dlg.result["high"],
        )

    def op_noise_impulse(self):
        if not self._ensure_image_loaded():
            return
        dlg = MultiFieldDialog(
            self.root,
            "Impulse Noise (Salt & Pepper)",
            [
                {"name": "pa", "label": "Probabilitas Pepper Pa (0..1):", "type": "float", "default": 0.03, "min": 0.0, "max": 1.0},
                {"name": "pb", "label": "Probabilitas Salt Pb (0..1):", "type": "float", "default": 0.03, "min": 0.0, "max": 1.0},
            ],
        )
        if dlg.result is None:
            return
        self._run_pipeline(
            f"Impulse Noise (Pa={dlg.result['pa']}, Pb={dlg.result['pb']})",
            noise.apply_impulse_noise,
            p_pepper=dlg.result["pa"],
            p_salt=dlg.result["pb"],
        )

    # =================================================================
    # E. MENU EDGE DETECTION
    # =================================================================
    def op_edge_sobel(self):
        self._run_pipeline("Edge Detection - Sobel", edge_detection.apply_sobel)

    def op_edge_prewitt(self):
        self._run_pipeline("Edge Detection - Prewitt", edge_detection.apply_prewitt)

    def op_edge_robert(self):
        self._run_pipeline("Edge Detection - Robert", edge_detection.apply_robert)

    def op_edge_laplacian(self):
        if not self._ensure_image_loaded():
            return
        dlg = MultiFieldDialog(
            self.root,
            "Edge Detection - Laplacian",
            [{"name": "mode", "label": "Kernel Laplacian:", "type": "choice", "default": "4-Tetangga", "choices": ["4-Tetangga", "8-Tetangga"]}],
        )
        if dlg.result is None:
            return
        m = "8" if dlg.result["mode"].startswith("8") else "4"
        self._run_pipeline(f"Edge Detection - Laplacian ({dlg.result['mode']})", edge_detection.apply_laplacian, neighbor_mode=m)

    def op_edge_log(self):
        if not self._ensure_image_loaded():
            return
        dlg = MultiFieldDialog(
            self.root,
            "Edge Detection - Laplacian of Gaussian (LoG)",
            [
                {"name": "mode", "label": "Metode LoG:", "type": "choice", "default": "Mexican Hat 5x5", "choices": ["Mexican Hat 5x5", "Gaussian + Laplacian"]},
                {"name": "sigma", "label": "Gaussian Sigma (khusus mode ke-2):", "type": "float", "default": 1.4, "min": 0.3, "max": 5.0},
            ],
        )
        if dlg.result is None:
            return
        mode_key = "mexican_hat" if "Mexican" in dlg.result["mode"] else "gaussian_laplacian"
        self._run_pipeline(f"Edge Detection - LoG ({dlg.result['mode']})", edge_detection.apply_log, mode=mode_key, sigma=dlg.result["sigma"])

    def op_edge_canny(self):
        if not self._ensure_image_loaded():
            return
        dlg = MultiFieldDialog(
            self.root,
            "Edge Detection - Canny",
            [
                {"name": "sigma", "label": "Gaussian Kernel SD (σ):", "type": "float", "default": 1.4, "min": 0.3, "max": 5.0},
                {"name": "t1", "label": "Upper Threshold (T1):", "type": "float", "default": 100.0, "min": 1.0, "max": 500.0},
                {"name": "t2", "label": "Lower Threshold (T2):", "type": "float", "default": 40.0, "min": 0.0, "max": 500.0},
            ],
        )
        if dlg.result is None:
            return
        self._run_pipeline(
            f"Edge Detection - Canny (σ={dlg.result['sigma']}, T1={dlg.result['t1']}, T2={dlg.result['t2']})",
            edge_detection.apply_canny,
            sigma=dlg.result["sigma"],
            upper_threshold=dlg.result["t1"],
            lower_threshold=dlg.result["t2"],
        )

    def op_edge_compass(self):
        self._run_pipeline("Edge Detection - Compass (8 Arah)", edge_detection.apply_compass)

    # =================================================================
    # F. MENU SEGMENTATION
    # =================================================================
    def op_seg_otsu(self):
        self._run_pipeline("Segmentation - Global Thresholding (Otsu)", segmentation.apply_global_otsu_threshold)

    def op_seg_adaptive(self):
        if not self._ensure_image_loaded():
            return
        dlg = MultiFieldDialog(
            self.root,
            "Segmentation - Adaptive Thresholding",
            [
                {"name": "bs", "label": "Ukuran Blok (ganjil >= 3):", "type": "int", "default": 15, "min": 3, "max": 101},
                {"name": "c", "label": "Konstanta C:", "type": "float", "default": 5.0},
            ],
        )
        if dlg.result is None:
            return
        self._run_pipeline(
            f"Adaptive Thresholding (B={dlg.result['bs']}, C={dlg.result['c']})",
            segmentation.apply_adaptive_threshold,
            block_size=dlg.result["bs"],
            c_const=dlg.result["c"],
        )

    def op_seg_contour(self):
        if not self._ensure_image_loaded():
            return
        dlg = MultiFieldDialog(
            self.root,
            "Segmentation - Contour Detection",
            [{"name": "overlay", "label": "Tampilkan kontur overlay di atas citra asli", "type": "bool", "default": True}],
        )
        if dlg.result is None:
            return
        self._run_pipeline(
            "Segmentation - Contour Detection",
            segmentation.apply_contour_detection,
            overlay=dlg.result["overlay"],
        )

    def op_seg_hough_line(self):
        if not self._ensure_image_loaded():
            return
        dlg = MultiFieldDialog(
            self.root,
            "Segmentation - Hough Line Transform",
            [
                {"name": "n", "label": "Maksimal Jumlah Garis:", "type": "int", "default": 10, "min": 1, "max": 50},
                {"name": "ratio", "label": "Rasio Ambang Vote (0.1 - 0.95):", "type": "float", "default": 0.45, "min": 0.1, "max": 0.95},
            ],
        )
        if dlg.result is None:
            return
        self._run_pipeline(
            f"Hough Line (N={dlg.result['n']})",
            segmentation.apply_hough_line,
            num_lines=dlg.result["n"],
            threshold_ratio=dlg.result["ratio"],
        )

    def op_seg_hough_circle(self):
        if not self._ensure_image_loaded():
            return
        dlg = MultiFieldDialog(
            self.root,
            "Segmentation - Hough Circle Transform",
            [
                {"name": "rmin", "label": "Radius Minimal (px):", "type": "int", "default": 15, "min": 5, "max": 300},
                {"name": "rmax", "label": "Radius Maksimal (px):", "type": "int", "default": 60, "min": 6, "max": 400},
                {"name": "nc", "label": "Maksimal Jumlah Lingkaran:", "type": "int", "default": 5, "min": 1, "max": 20},
            ],
        )
        if dlg.result is None:
            return
        self._run_pipeline(
            f"Hough Circle (R={dlg.result['rmin']}..{dlg.result['rmax']})",
            segmentation.apply_hough_circle,
            r_min=dlg.result["rmin"],
            r_max=dlg.result["rmax"],
            num_circles=dlg.result["nc"],
        )

    def op_seg_thinning(self):
        if not self._ensure_image_loaded():
            return
        dlg = MultiFieldDialog(
            self.root,
            "Segmentation - Thinning (Skeletonization)",
            [{"name": "inv", "label": "Objek berwarna gelap di atas latar terang (Invert)", "type": "bool", "default": False}],
        )
        if dlg.result is None:
            return
        self._run_pipeline(
            "Segmentation - Thinning (Zhang-Suen)",
            segmentation.apply_thinning,
            invert_fg=dlg.result["inv"],
        )

    # =================================================================
    # G. MENU COMPRESSION (BONUS / OPSIONAL)
    # =================================================================
    def op_comp_huffman(self):
        metrics = self._run_pipeline("Compression - Huffman Coding", compression.apply_huffman_compression)
        if metrics:
            show_compression_report(self.root, metrics)

    def op_comp_rle(self):
        metrics = self._run_pipeline("Compression - Run-Length Encoding", compression.apply_rle_compression)
        if metrics:
            show_compression_report(self.root, metrics)

    def op_comp_quantization(self):
        if not self._ensure_image_loaded():
            return
        dlg = SliderDialog(self.root, "Compression - Kuantisasi Seragam", "Kedalaman Bit Target (1 - 7 bit):", 1, 7, 3, is_int=True)
        if dlg.result is None:
            return
        metrics = self._run_pipeline(
            f"Compression - Quantization ({dlg.result}-bit)",
            compression.apply_quantization_compression,
            bits=dlg.result,
        )
        if metrics:
            show_compression_report(self.root, metrics)

    # =================================================================
    # H. MENU ABOUT
    # =================================================================
    def show_about(self):
        show_about_dialog(self.root, APP_CONFIG)

    def open_github_link(self):
        webbrowser.open(APP_CONFIG["github_url"])

    def open_youtube_link(self):
        webbrowser.open(APP_CONFIG["youtube_url"])
