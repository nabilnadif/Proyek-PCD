"""
Entry Point Aplikasi Pengolahan Citra Digital (PCD) Desktop.
Menjalankan inisialisasi Tkinter root window dan mainloop().
"""
import tkinter as tk
from ui.app_window import ImageProcessingApp


def main():
    root = tk.Tk()
    ImageProcessingApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
