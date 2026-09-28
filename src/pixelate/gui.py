"""Small Tk front end: ``pixelate-gui [IMAGE]``."""

from __future__ import annotations

import queue
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk

from .abstraction import Params, Progress, pixelate, resize_baseline

PREVIEW = 400


def _fit(img: Image.Image, box: int = PREVIEW, resample=Image.Resampling.NEAREST) -> Image.Image:
    scale = box / max(img.width, img.height)
    return img.resize((max(1, round(img.width * scale)), max(1, round(img.height * scale))), resample)


class App:
    def __init__(self, root: tk.Tk, path: str | None = None) -> None:
        self.root = root
        self.image: Image.Image | None = None
        self.result: Image.Image | None = None
        self.events: queue.Queue = queue.Queue()
        self.worker: threading.Thread | None = None
        root.title("Pixelated Image Abstraction")

        self.canvas = tk.Canvas(root, width=2 * PREVIEW + 30, height=PREVIEW + 60, bg="white")
        self.canvas.grid(row=0, column=0, padx=10, pady=10)

        panel = ttk.Frame(root, padding=10)
        panel.grid(row=0, column=1, sticky="n")
        self.width = tk.IntVar(value=64)
        self.height = tk.IntVar(value=64)
        self.keep_aspect = tk.BooleanVar(value=True)
        self.colors = tk.IntVar(value=8)
        self.method = tk.StringVar(value="paper")
        self.status = tk.StringVar(value="Load an image to start")

        ttk.Button(panel, text="Load image…", command=self.load).grid(row=0, column=0, columnspan=2, sticky="ew")
        rows = [("Width", ttk.Spinbox(panel, from_=1, to=1024, textvariable=self.width, width=8)),
                ("Height", ttk.Spinbox(panel, from_=1, to=1024, textvariable=self.height, width=8)),
                ("Colours", ttk.Spinbox(panel, from_=1, to=256, textvariable=self.colors, width=8)),
                ("Method", ttk.Combobox(panel, textvariable=self.method, values=["paper", "nearest", "box"],
                                        state="readonly", width=8))]
        for i, (label, widget) in enumerate(rows, start=1):
            ttk.Label(panel, text=label).grid(row=i, column=0, sticky="w", pady=2)
            widget.grid(row=i, column=1, sticky="ew", pady=2)
        ttk.Checkbutton(panel, text="Keep aspect ratio", variable=self.keep_aspect).grid(
            row=5, column=0, columnspan=2, sticky="w")
        self.width.trace_add("write", lambda *_: self._sync_height())

        self.run_btn = ttk.Button(panel, text="Pixelate", command=self.run)
        self.run_btn.grid(row=6, column=0, columnspan=2, sticky="ew", pady=(12, 2))
        ttk.Button(panel, text="Save result…", command=self.save).grid(row=7, column=0, columnspan=2, sticky="ew")
        ttk.Label(panel, textvariable=self.status, wraplength=180).grid(row=8, column=0, columnspan=2, pady=10)

        if path:
            self.open(path)

    def _sync_height(self) -> None:
        if not (self.keep_aspect.get() and self.image):
            return
        try:
            w = self.width.get()
        except tk.TclError:
            return
        if w > 0:
            self.height.set(max(1, round(self.image.height * w / self.image.width)))

    def load(self) -> None:
        path = filedialog.askopenfilename(title="Select an image")
        if path:
            self.open(path)

    def open(self, path: str) -> None:
        try:
            self.image = Image.open(path)
            self.image.load()
        except OSError as exc:
            messagebox.showerror("Cannot open image", str(exc))
            return
        self.result = None
        self._sync_height()
        self._show(0, self.image, Image.Resampling.LANCZOS)
        self.status.set(f"{self.image.width}x{self.image.height}")

    def _show(self, slot: int, img: Image.Image, resample=Image.Resampling.NEAREST) -> None:
        photo = ImageTk.PhotoImage(_fit(img, resample=resample))
        setattr(self, f"_photo{slot}", photo)  # keep a reference or Tk drops the image
        self.canvas.delete(f"img{slot}")
        self.canvas.create_image(10 + slot * (PREVIEW + 10), 10, anchor="nw", image=photo, tags=f"img{slot}")

    def _show_palette(self, colors) -> None:
        self.canvas.delete("palette")
        for i, c in enumerate(colors[:40]):
            x = 10 + i * 20
            self.canvas.create_rectangle(x, PREVIEW + 25, x + 18, PREVIEW + 43, tags="palette",
                                         fill="#%02x%02x%02x" % tuple(int(v) for v in c[:3]))

    def run(self) -> None:
        if self.image is None or (self.worker and self.worker.is_alive()):
            return
        try:
            w, h, k = self.width.get(), self.height.get(), self.colors.get()
        except tk.TclError:
            messagebox.showerror("Invalid settings", "Width, height and colours must be numbers")
            return
        image, method = self.image, self.method.get()

        def work() -> None:
            try:
                if method == "paper":
                    res = pixelate(image, w, h, Params(colors=k),
                                   progress=lambda pr: self.events.put(("progress", pr)))
                    self.events.put(("done", (res.image, res.palette)))
                else:
                    out = resize_baseline(image, w, h, k, method)
                    self.events.put(("done", (out, sorted(set(out.convert("RGB").getdata())))))
            except Exception as exc:  # surface any failure in the UI instead of a dead thread
                self.events.put(("error", exc))

        self.run_btn.state(["disabled"])
        self.status.set("Working…")
        self.worker = threading.Thread(target=work, daemon=True)
        self.worker.start()
        self.root.after(100, self._poll)

    def _poll(self) -> None:
        try:
            while True:
                kind, payload = self.events.get_nowait()
                if kind == "progress":
                    pr: Progress = payload
                    self.status.set(f"Iteration {pr.iteration}\nT = {pr.temperature:.1f}\n"
                                    f"{pr.palette_size} colours")
                elif kind == "done":
                    self.result, palette = payload
                    self._show(1, self.result)
                    self._show_palette(palette)
                    self.status.set(f"Done: {self.result.width}x{self.result.height}, {len(palette)} colours")
                    self.run_btn.state(["!disabled"])
                    return
                else:
                    self.run_btn.state(["!disabled"])
                    self.status.set("Failed")
                    messagebox.showerror("Pixelation failed", str(payload))
                    return
        except queue.Empty:
            pass
        self.root.after(100, self._poll)

    def save(self) -> None:
        if self.result is None:
            return
        path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png")])
        if path:
            self.result.save(path)
            self.status.set(f"Saved {path}")


def main() -> None:
    root = tk.Tk()
    App(root, sys.argv[1] if len(sys.argv) > 1 else None)
    root.mainloop()


if __name__ == "__main__":
    main()
