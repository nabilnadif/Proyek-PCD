"""
Modul ui/dialogs.py
Kumpulan jendela dialog pop-up input parameter (skalar, sudut, D0, threshold,
crop, kernel konvolusi 3x3, histogram, laporan kompresi, dan About).
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
from PIL import Image


class BaseParamDialog(tk.Toplevel):
    """Kelas dasar modal dialog dengan validasi input dan tombol OK / Batal."""

    def __init__(self, parent, title: str):
        super().__init__(parent)
        self.title(title)
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()
        self.result = None

        self.container = ttk.Frame(self, padding=14)
        self.container.pack(fill=tk.BOTH, expand=True)

        self.body = ttk.Frame(self.container)
        self.body.pack(fill=tk.BOTH, expand=True)

        self._build_body(self.body)

        btn_frame = ttk.Frame(self.container)
        btn_frame.pack(fill=tk.X, pady=(12, 0))
        ttk.Button(btn_frame, text="Batal", command=self._on_cancel).pack(side=tk.RIGHT, padx=(6, 0))
        ttk.Button(btn_frame, text="Terapkan", command=self._on_ok).pack(side=tk.RIGHT)

        self.bind("<Return>", lambda _e: self._on_ok())
        self.bind("<Escape>", lambda _e: self._on_cancel())

        self.update_idletasks()
        self._center_to_parent(parent)
        self.wait_window(self)

    def _center_to_parent(self, parent):
        px = parent.winfo_rootx() + (parent.winfo_width() - self.winfo_width()) // 2
        py = parent.winfo_rooty() + (parent.winfo_height() - self.winfo_height()) // 2
        self.geometry(f"+{max(0, px)}+{max(0, py)}")

    def _build_body(self, master):
        raise NotImplementedError

    def _validate(self):
        raise NotImplementedError

    def _on_ok(self):
        try:
            self.result = self._validate()
            self.destroy()
        except Exception as exc:
            messagebox.showerror("Input Tidak Valid", str(exc), parent=self)

    def _on_cancel(self):
        self.result = None
        self.destroy()


class MultiFieldDialog(BaseParamDialog):
    """
    Dialog fleksibel untuk berbagai kombinasi input angka, dropdown, dan checkbox.
    fields: list of dict dengan keys:
      - name: str
      - label: str
      - type: 'int' | 'float' | 'choice' | 'bool'
      - default: nilai awal
      - choices: list (khusus type='choice')
      - min, max: batas opsional (int/float)
    """

    def __init__(self, parent, title: str, fields: list, subtitle: str = None):
        self._fields = fields
        self._subtitle = subtitle
        self._vars = {}
        super().__init__(parent, title)

    def _build_body(self, master):
        row = 0
        if self._subtitle:
            ttk.Label(master, text=self._subtitle, wraplength=340, foreground="#444444").grid(
                row=row, column=0, columnspan=2, sticky=tk.W, pady=(0, 8)
            )
            row += 1

        for spec in self._fields:
            name = spec["name"]
            ftype = spec["type"]
            default = spec.get("default", "")

            if ftype == "bool":
                var = tk.BooleanVar(value=bool(default))
                cb = ttk.Checkbutton(master, text=spec["label"], variable=var)
                cb.grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=3)
                self._vars[name] = (ftype, var, spec)
            else:
                ttk.Label(master, text=spec["label"]).grid(row=row, column=0, sticky=tk.W, pady=4, padx=(0, 10))
                if ftype == "choice":
                    var = tk.StringVar(value=str(default))
                    widget = ttk.Combobox(
                        master, textvariable=var, values=spec["choices"], state="readonly", width=22
                    )
                    widget.grid(row=row, column=1, sticky=tk.EW, pady=4)
                else:
                    var = tk.StringVar(value=str(default))
                    widget = ttk.Entry(master, textvariable=var, width=24)
                    widget.grid(row=row, column=1, sticky=tk.EW, pady=4)
                    if row == (1 if self._subtitle else 0):
                        widget.focus_set()
                self._vars[name] = (ftype, var, spec)
            row += 1

        master.columnconfigure(1, weight=1)

    def _validate(self):
        out = {}
        for name, (ftype, var, spec) in self._vars.items():
            if ftype == "bool":
                out[name] = bool(var.get())
            elif ftype == "choice":
                out[name] = var.get()
            elif ftype == "int":
                val = int(var.get().strip())
                if "min" in spec and val < spec["min"]:
                    raise ValueError(f"{spec['label']} minimal {spec['min']}.")
                if "max" in spec and val > spec["max"]:
                    raise ValueError(f"{spec['label']} maksimal {spec['max']}.")
                out[name] = val
            elif ftype == "float":
                val = float(var.get().strip().replace(",", "."))
                if "min" in spec and val < spec["min"]:
                    raise ValueError(f"{spec['label']} minimal {spec['min']}.")
                if "max" in spec and val > spec["max"]:
                    raise ValueError(f"{spec['label']} maksimal {spec['max']}.")
                out[name] = val
        return out


class SliderDialog(BaseParamDialog):
    """Dialog input tunggal dengan slider (Scale) + kotak angka sinkron."""

    def __init__(self, parent, title: str, label: str, min_val: float, max_val: float, default: float, is_int: bool = True):
        self._label = label
        self._min = min_val
        self._max = max_val
        self._default = default
        self._is_int = is_int
        super().__init__(parent, title)

    def _build_body(self, master):
        ttk.Label(master, text=self._label).pack(anchor=tk.W, pady=(0, 6))
        self.val_var = tk.DoubleVar(value=self._default)
        self.entry_var = tk.StringVar(value=str(int(self._default) if self._is_int else self._default))

        row = ttk.Frame(master)
        row.pack(fill=tk.X)

        def on_scale(v):
            num = int(round(float(v))) if self._is_int else round(float(v), 2)
            self.entry_var.set(str(num))

        scale = ttk.Scale(row, from_=self._min, to=self._max, variable=self.val_var, command=on_scale, length=240)
        scale.pack(side=tk.LEFT, padx=(0, 10))

        entry = ttk.Entry(row, textvariable=self.entry_var, width=8)
        entry.pack(side=tk.LEFT)
        entry.focus_set()

    def _validate(self):
        raw = self.entry_var.get().strip().replace(",", ".")
        val = int(round(float(raw))) if self._is_int else float(raw)
        if val < self._min or val > self._max:
            raise ValueError(f"Nilai harus berada pada rentang [{self._min}, {self._max}].")
        return val


class ArithmeticDialog(BaseParamDialog):
    """Dialog pemilihan mode Aritmatika: Skalar c atau Citra Kedua B."""

    def __init__(self, parent, op_label: str):
        self._op_label = op_label
        self._second_img = None
        super().__init__(parent, f"Arithmetic - {op_label}")

    def _build_body(self, master):
        self.mode_var = tk.StringVar(value="scalar")

        ttk.Radiobutton(
            master, text="Mode Skalar (konstanta c)", variable=self.mode_var, value="scalar", command=self._toggle_mode
        ).grid(row=0, column=0, columnspan=2, sticky=tk.W, pady=2)

        ttk.Label(master, text="Nilai Skalar (c):").grid(row=1, column=0, sticky=tk.W, padx=(20, 8), pady=4)
        default_val = "1.2" if self._op_label in ("Multiply (*)", "Divide (/)") else "30"
        self.scalar_var = tk.StringVar(value=default_val)
        self.scalar_entry = ttk.Entry(master, textvariable=self.scalar_var, width=18)
        self.scalar_entry.grid(row=1, column=1, sticky=tk.EW, pady=4)

        ttk.Separator(master, orient=tk.HORIZONTAL).grid(row=2, column=0, columnspan=2, sticky=tk.EW, pady=8)

        ttk.Radiobutton(
            master, text="Mode 2-Citra (Citra Kedua B)", variable=self.mode_var, value="image", command=self._toggle_mode
        ).grid(row=3, column=0, columnspan=2, sticky=tk.W, pady=2)

        self.path_var = tk.StringVar(value="(Belum ada citra kedua dipilih)")
        self.browse_btn = ttk.Button(master, text="Pilih Citra B...", command=self._browse_image)
        self.browse_btn.grid(row=4, column=0, sticky=tk.W, padx=(20, 8), pady=4)
        ttk.Label(master, textvariable=self.path_var, width=28).grid(row=4, column=1, sticky=tk.W, pady=4)

        self._toggle_mode()

    def _toggle_mode(self):
        is_scalar = self.mode_var.get() == "scalar"
        self.scalar_entry.configure(state="normal" if is_scalar else "disabled")
        self.browse_btn.configure(state="disabled" if is_scalar else "normal")

    def _browse_image(self):
        path = filedialog.askopenfilename(
            parent=self,
            title="Pilih Citra Kedua (B)",
            filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff")],
        )
        if path:
            img = Image.open(path)
            self._second_img = np.asarray(img, dtype=np.float64)
            import os
            self.path_var.set(os.path.basename(path))

    def _validate(self):
        mode = self.mode_var.get()
        if mode == "scalar":
            val = float(self.scalar_var.get().strip().replace(",", "."))
            if self._op_label == "Divide (/)" and abs(val) < 1e-12:
                raise ZeroDivisionError("Pembagi skalar c tidak boleh bernilai 0.")
            return {"mode": "scalar", "scalar": val, "second_img": None}
        else:
            if self._second_img is None:
                raise ValueError("Silakan pilih file Citra Kedua (B) terlebih dahulu.")
            return {"mode": "image", "scalar": 0.0, "second_img": self._second_img}


class ConvolutionDialog(BaseParamDialog):
    """Dialog input matriks kernel 3x3 custom dengan preset populer."""

    PRESETS = {
        "Identity": [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
        "Box Blur (1/9)": [[0.111, 0.111, 0.111], [0.111, 0.111, 0.111], [0.111, 0.111, 0.111]],
        "Sharpen": [[0, -1, 0], [-1, 5, -1], [0, -1, 0]],
        "Edge Enhance": [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]],
        "Emboss": [[-2, -1, 0], [-1, 1, 1], [0, 1, 2]],
        "Gaussian 3x3": [[0.0625, 0.125, 0.0625], [0.125, 0.25, 0.125], [0.0625, 0.125, 0.0625]],
    }

    def __init__(self, parent):
        self._entries = []
        super().__init__(parent, "Convolution - Matriks Kernel 3x3")

    def _build_body(self, master):
        top = ttk.Frame(master)
        top.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(top, text="Preset Kernel:").pack(side=tk.LEFT, padx=(0, 8))

        self.preset_var = tk.StringVar(value="Sharpen")
        cb = ttk.Combobox(top, textvariable=self.preset_var, values=list(self.PRESETS.keys()), state="readonly", width=18)
        cb.pack(side=tk.LEFT)
        cb.bind("<<ComboboxSelected>>", lambda _e: self._load_preset())

        grid_frame = ttk.LabelFrame(master, text="Elemen Kernel 3x3", padding=8)
        grid_frame.pack()

        self._entries = []
        for r in range(3):
            row_vars = []
            for c in range(3):
                var = tk.StringVar()
                ent = ttk.Entry(grid_frame, textvariable=var, width=8, justify=tk.CENTER)
                ent.grid(row=r, column=c, padx=4, pady=4)
                row_vars.append(var)
            self._entries.append(row_vars)

        self._load_preset()

    def _load_preset(self):
        mat = self.PRESETS[self.preset_var.get()]
        for r in range(3):
            for c in range(3):
                self._entries[r][c].set(str(mat[r][c]))

    def _validate(self):
        mat = np.zeros((3, 3), dtype=np.float64)
        for r in range(3):
            for c in range(3):
                mat[r, c] = float(self._entries[r][c].get().strip().replace(",", "."))
        return mat


class ContrastDialog(BaseParamDialog):
    """Dialog Contrast dengan pilihan Mode Scaling (G, P) atau Stretching (r_min, r_max)."""

    def __init__(self, parent, actual_min: int, actual_max: int):
        self._actual_min = actual_min
        self._actual_max = actual_max
        super().__init__(parent, "Contrast Enhancement")

    def _build_body(self, master):
        self.mode_var = tk.StringVar(value="scaling")

        ttk.Radiobutton(
            master, text="Contrast Scaling: f_o = G * (f_i - P) + P",
            variable=self.mode_var, value="scaling", command=self._toggle
        ).grid(row=0, column=0, columnspan=2, sticky=tk.W, pady=2)

        ttk.Label(master, text="Gain (G):").grid(row=1, column=0, sticky=tk.W, padx=(20, 8), pady=3)
        self.gain_var = tk.StringVar(value="1.5")
        self.gain_ent = ttk.Entry(master, textvariable=self.gain_var, width=14)
        self.gain_ent.grid(row=1, column=1, sticky=tk.EW, pady=3)

        ttk.Label(master, text="Pivot (P):").grid(row=2, column=0, sticky=tk.W, padx=(20, 8), pady=3)
        self.pivot_var = tk.StringVar(value="128")
        self.pivot_ent = ttk.Entry(master, textvariable=self.pivot_var, width=14)
        self.pivot_ent.grid(row=2, column=1, sticky=tk.EW, pady=3)

        ttk.Separator(master, orient=tk.HORIZONTAL).grid(row=3, column=0, columnspan=2, sticky=tk.EW, pady=8)

        ttk.Radiobutton(
            master, text="Contrast Stretching: ((r - r_min) / (r_max - r_min)) * 255",
            variable=self.mode_var, value="stretching", command=self._toggle
        ).grid(row=4, column=0, columnspan=2, sticky=tk.W, pady=2)

        ttk.Label(master, text="r_min (0-255):").grid(row=5, column=0, sticky=tk.W, padx=(20, 8), pady=3)
        self.rmin_var = tk.StringVar(value=str(self._actual_min))
        self.rmin_ent = ttk.Entry(master, textvariable=self.rmin_var, width=14)
        self.rmin_ent.grid(row=5, column=1, sticky=tk.EW, pady=3)

        ttk.Label(master, text="r_max (0-255):").grid(row=6, column=0, sticky=tk.W, padx=(20, 8), pady=3)
        self.rmax_var = tk.StringVar(value=str(self._actual_max))
        self.rmax_ent = ttk.Entry(master, textvariable=self.rmax_var, width=14)
        self.rmax_ent.grid(row=6, column=1, sticky=tk.EW, pady=3)

        self._toggle()

    def _toggle(self):
        is_scale = self.mode_var.get() == "scaling"
        self.gain_ent.configure(state="normal" if is_scale else "disabled")
        self.pivot_ent.configure(state="normal" if is_scale else "disabled")
        self.rmin_ent.configure(state="disabled" if is_scale else "normal")
        self.rmax_ent.configure(state="disabled" if is_scale else "normal")

    def _validate(self):
        mode = self.mode_var.get()
        if mode == "scaling":
            g = float(self.gain_var.get().strip().replace(",", "."))
            p = float(self.pivot_var.get().strip().replace(",", "."))
            return {"mode": "scaling", "gain": g, "pivot": p}
        else:
            rmin = float(self.rmin_var.get().strip().replace(",", "."))
            rmax = float(self.rmax_var.get().strip().replace(",", "."))
            if rmax <= rmin:
                raise ValueError("Nilai r_max harus lebih besar dari r_min.")
            return {"mode": "stretching", "r_min": rmin, "r_max": rmax}


def show_histogram_window(parent, original_img: Image.Image, processed_img: Image.Image):
    """Menampilkan perbandingan histogram Original vs Processed Image (Matplotlib / Canvas fallback)."""
    win = tk.Toplevel(parent)
    win.title("Visualisasi Histogram Citra")
    win.geometry("860x420")
    win.transient(parent)

    orig_arr = np.asarray(original_img)
    proc_arr = np.asarray(processed_img)

    try:
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

        fig = Figure(figsize=(8.4, 3.8), dpi=100)
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)

        def _plot_hist(ax, arr, title):
            ax.set_title(title, fontsize=10)
            ax.set_xlabel("Intensitas (0-255)")
            ax.set_ylabel("Jumlah Piksel")
            ax.set_xlim(0, 255)
            ax.grid(True, linestyle="--", alpha=0.35)
            if arr.ndim == 2:
                ax.hist(arr.ravel(), bins=256, range=(0, 256), color="#333333", alpha=0.85)
            else:
                for idx, col in enumerate(("red", "green", "blue")):
                    ax.hist(arr[:, :, idx].ravel(), bins=256, range=(0, 256), color=col, alpha=0.45, label=col.upper())
                ax.legend(fontsize=8)

        _plot_hist(ax1, orig_arr, "Histogram Original Image")
        _plot_hist(ax2, proc_arr, "Histogram Processed Image")
        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    except ImportError:
        # Fallback murni Tkinter Canvas bila matplotlib belum terinstal
        frame = ttk.Frame(win, padding=10)
        frame.pack(fill=tk.BOTH, expand=True)
        for idx, (arr, title) in enumerate([(orig_arr, "Original Image"), (proc_arr, "Processed Image")]):
            sub = ttk.LabelFrame(frame, text=title, padding=6)
            sub.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=6)
            c = tk.Canvas(sub, bg="white", width=360, height=280)
            c.pack(fill=tk.BOTH, expand=True)
            gray = arr if arr.ndim == 2 else arr[:, :, :3].mean(axis=2)
            hist, _ = np.histogram(gray.ravel(), bins=256, range=(0, 256))
            hmax = max(1, int(hist.max()))
            for x in range(256):
                bar_h = int((hist[x] / hmax) * 240)
                x_pos = 30 + int(x * 1.2)
                c.create_line(x_pos, 260, x_pos, 260 - bar_h, fill="#2c3e50")


def show_compression_report(parent, metrics: dict):
    """Menampilkan hasil perhitungan Rasio Kompresi, BPP, MSE, dan PSNR."""
    psnr_str = "Infinity (Lossless)" if np.isinf(metrics["psnr"]) else f"{metrics['psnr']:.2f} dB"
    msg = (
        f"Metode Kompresi  : {metrics['method']}\n"
        f"Ukuran Asli      : {metrics['original_bits']:,} bit\n"
        f"Ukuran Terkompres: {metrics['compressed_bits']:,} bit\n"
        f"Compression Ratio: {metrics['compression_ratio']:.3f} : 1\n"
        f"Bit Per Pixel    : {metrics['bpp']:.3f} bpp\n"
        f"Penghematan Ruang: {metrics['space_saving_pct']:.2f}%\n"
        f"MSE              : {metrics['mse']:.4f}\n"
        f"PSNR             : {psnr_str}"
    )
    messagebox.showinfo("Laporan Evaluasi Kompresi Citra", msg, parent=parent)


def show_about_dialog(parent, app_info: dict):
    """Menampilkan jendela pop-up Info Tim Developer sesuai blueprint."""
    win = tk.Toplevel(parent)
    win.title("Info Tim Developer")
    win.resizable(False, False)
    win.transient(parent)
    win.grab_set()

    frame = ttk.Frame(win, padding=18)
    frame.pack(fill=tk.BOTH, expand=True)

    ttk.Label(frame, text=app_info["app_name"], font=("Segoe UI", 13, "bold")).pack(anchor=tk.W)
    ttk.Label(frame, text=app_info["copyright"], foreground="#555555").pack(anchor=tk.W, pady=(2, 10))

    ttk.Separator(frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=6)

    ttk.Label(frame, text=f"Mata Kuliah : {app_info['course']}").pack(anchor=tk.W, pady=2)
    ttk.Label(frame, text=f"Dosen Pengampu : {app_info['lecturer']}").pack(anchor=tk.W, pady=2)
    ttk.Label(frame, text=f"Institusi : {app_info['institution']}").pack(anchor=tk.W, pady=2)

    ttk.Label(frame, text="Anggota Tim:", font=("Segoe UI", 10, "bold")).pack(anchor=tk.W, pady=(10, 4))
    for member in app_info["members"]:
        ttk.Label(frame, text=f"  • {member}").pack(anchor=tk.W, pady=1)

    ttk.Separator(frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
    ttk.Button(frame, text="Tutup", command=win.destroy).pack(side=tk.RIGHT)

    win.update_idletasks()
    px = parent.winfo_rootx() + (parent.winfo_width() - win.winfo_width()) // 2
    py = parent.winfo_rooty() + (parent.winfo_height() - win.winfo_height()) // 2
    win.geometry(f"+{max(0, px)}+{max(0, py)}")
