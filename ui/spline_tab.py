import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from core.splines import make_grid, METHODS, SPLINE_COLORS
from ui.widgets import make_log, to_float


class SplineTab(ttk.Frame):
    MIN_INTERVALS = 1
    MAX_INTERVALS = 30

    def __init__(self, master):
        super().__init__(master)
        left = ttk.Frame(self, padding=10)
        left.pack(side="left", fill="y")
        right = ttk.Frame(self)
        right.pack(side="right", fill="both", expand=True)

        ttk.Label(left, text="Побудова B-сплайнів", font=("Segoe UI", 13, "bold")).pack(anchor="w", pady=(0, 4))
        ttk.Label(
            left,
            text="Сітка: x_i = a + i·h, i = 0…n, де n = (b − a)/h.\n"
                 "Лінійний (m=2) і кубічний (m=4) B-сплайни\n"
                 "будуються на окремих графіках. Сітку доповнено фіктивними\n"
                 "вузлами з обох боків; частини сплайнів за межами\n"
                 "[a; b] показано штрихпунктиром.",
            justify="left",
        ).pack(anchor="w", pady=(0, 8))

        grid = ttk.Frame(left)
        grid.pack(anchor="w", fill="x")
        self.a_var = tk.StringVar(value="0")
        self.b_var = tk.StringVar(value="8")
        self.h_var = tk.StringVar(value="1")
        fields = [
            ("Ліва межа [a]:", self.a_var),
            ("Права межа [b]:", self.b_var),
            ("Крок розбиття [h > 0]:", self.h_var),
        ]
        for r, (text, var) in enumerate(fields):
            ttk.Label(grid, text=text).grid(row=r, column=0, sticky="w", pady=3, padx=(0, 8))
            ttk.Entry(grid, textvariable=var, width=14).grid(row=r, column=1, sticky="w", pady=3)

        self.info_label = ttk.Label(left, foreground="#555", wraplength=380, justify="left")
        self.info_label.pack(anchor="w", pady=(6, 10))
        for var in (self.a_var, self.b_var, self.h_var):
            var.trace_add("write", lambda *_: self.update_info())

        ttk.Button(left, text="🚀 Побудувати B-сплайни", command=self.build).pack(fill="x", ipady=6, pady=(0, 10))

        ttk.Label(left, text="Протокол обчислень:").pack(anchor="w")
        log_frame, self.log = make_log(left, width=52, height=20)
        log_frame.pack(fill="both", expand=True)

        # два графіки поряд: лінійний і кубічний
        right.columnconfigure(0, weight=1, uniform="plots")
        right.columnconfigure(1, weight=1, uniform="plots")
        right.rowconfigure(0, weight=1)
        self.plots = {}
        for col, key in enumerate(("linear", "cubic")):
            cell = ttk.Frame(right)
            cell.grid(row=0, column=col, sticky="nsew")
            figure = Figure(figsize=(5, 6), tight_layout=True)
            ax = figure.add_subplot(111)
            canvas = FigureCanvasTkAgg(figure, master=cell)
            toolbar = NavigationToolbar2Tk(canvas, cell)
            toolbar.update()
            toolbar.pack(side="bottom", fill="x")
            canvas.get_tk_widget().pack(side="top", fill="both", expand=True)
            self.plots[key] = (figure, ax, canvas)

        self.update_info()
        self.build()

    def read_grid(self):
        """Читає a, b, h, перевіряє їх і повертає (a, b, h, n)."""
        a = to_float(self.a_var.get(), "a")
        b = to_float(self.b_var.get(), "b")
        h = to_float(self.h_var.get(), "h")
        if not a < b:
            raise ValueError("Права межа b має бути більшою за ліву межу a.")
        if h <= 0:
            raise ValueError("Крок h має бути більшим за нуль.")
        ratio = (b - a) / h
        n = int(round(ratio))
        if not np.isclose(ratio, n, rtol=1e-9, atol=1e-8):
            raise ValueError(
                f"Крок h = {h:g} не ділить проміжок [{a:g}; {b:g}] на цілу кількість "
                f"частин: (b − a)/h = {ratio:.8g}. Змініть h або межі."
            )
        if not self.MIN_INTERVALS <= n <= self.MAX_INTERVALS:
            raise ValueError(
                f"Кількість проміжків n = {n}. Вона має бути від {self.MIN_INTERVALS} "
                f"до {self.MAX_INTERVALS}."
            )
        return a, b, h, n

    def update_info(self):
        try:
            a, b, h, n = self.read_grid()
            self.info_label.config(
                text=f"Проміжок: [{a:g}; {b:g}]\nКрок h = {h:g}\nКількість проміжків n = {n} "
                     f"(вузлів: {n + 1})"
            )
        except ValueError as error:
            self.info_label.config(text=str(error))

    def write_log(self, text):
        self.log.delete("1.0", "end")
        self.log.insert("end", text)

    def build(self):
        try:
            a, b, h, n = self.read_grid()
            nodes = make_grid(a, h, n)
            for key in ("linear", "cubic"):
                self.draw(key, nodes, a, h, n)
            self.fill_log(nodes, a, b, h, n)
        except ValueError as error:
            messagebox.showwarning("Перевір вхідні дані", str(error))
        except Exception as error:
            messagebox.showerror("Помилка обчислення", f"Сталася помилка:\n{error}")

    def draw(self, key, nodes, x0, h, n):
        info = METHODS[key]
        func, width, tag, name = info["func"], info["width"], info["tag"], info["name"]
        ghost = width - 1                      # кількість фіктивних вузлів з кожного боку
        k_min, k_max = -ghost, n - 1           # усі сплайни, носій яких перетинає [a; b]
        cnt = k_max - k_min + 1
        a, b = nodes[0], nodes[-1]
        eps = 1e-9 * h
        ext = x0 + h * np.arange(-ghost, n + width, dtype=float)
        ext_idx = np.arange(-ghost, n + width)

        figure, ax, canvas = self.plots[key]
        ax.clear()

        ax.axvspan(ext[0], a, color="gray", alpha=0.12)
        ax.axvspan(b, ext[-1], color="gray", alpha=0.12)
        ax.plot([ext[0], ext[-1]], [0, 0], color="black", lw=3, zorder=2)

        for k in range(k_min, k_max + 1):
            color = SPLINE_COLORS[(k - k_min) % len(SPLINE_COLORS)]
            seg = np.linspace(x0 + k * h, x0 + (k + width) * h, 100 * width + 1)
            vals = func(seg, k, x0, h)
            label = f"$B_{{{tag},{k}}}$" if cnt <= 6 else None
            parts = [
                (seg <= a + eps, "-."),
                ((seg >= a - eps) & (seg <= b + eps), "-"),
                (seg >= b - eps, "-."),
            ]
            for mask, style in parts:
                if mask.sum() > 1:
                    ax.plot(seg[mask], vals[mask], style, lw=2.5, color=color, label=label, zorder=3)
                    label = None

        ax.axvline(a, color="black", lw=1.4, zorder=4)
        ax.axvline(b, color="black", lw=1.4, zorder=4)

        real = (ext_idx >= 0) & (ext_idx <= n)
        ax.plot(ext[real], np.zeros(real.sum()), "o", color="red", ms=6, zorder=5, label="вузли $x_i$")
        ax.plot(ext[~real], np.zeros((~real).sum()), "o", mfc="white", mec="red", ms=6, zorder=5, label="фіктивні вузли")
        for xn in ext:
            ax.axvline(xn, color="gray", ls=":", lw=0.8)
        ax.plot([], [], "-.", color="gray", lw=2, label="сплайни за межами $[a; b]$")

        for q in range(1, width):
            xv = x0 + q * h
            yv = float(func(xv, 0, x0, h))
            ax.plot(xv, yv, "s", color="black", ms=5, zorder=6)
            ax.annotate(
                f"{yv:.4f}", (xv, yv), textcoords="offset points",
                xytext=(0, 9), ha="center", fontsize=9, fontweight="bold"
            )

        peak = float(np.max(func(np.linspace(x0, x0 + width * h, 500), 0, x0, h)))
        ax.set_xticks(ext)
        ax.set_xticklabels([f"$x_{{{i}}}$\n{xv:.4g}" for i, xv in zip(ext_idx, ext)], fontsize=7)
        ax.set_ylim(-0.05 * peak, peak * 1.3)
        ax.set_title(
            f"{name}\n$B_{{{tag},{k_min}}}$ … $B_{{{tag},{k_max}}}$ ({cnt} шт.), "
            f"фіктивних вузлів: {ghost}+{ghost}", fontsize=10
        )
        ax.set_xlabel("$x$ (вузли сітки)")
        ax.set_ylabel(f"$B_{{{tag},k}}(x)$")
        ax.grid(True, alpha=0.3)
        ax.legend(ncol=2, fontsize=8, loc="upper left")
        figure.tight_layout()
        canvas.draw()

    def fill_log(self, nodes, a, b, h, n):
        lines = [
            "=== ПАРАМЕТРИ СІТКИ ===",
            f"Проміжок:             [{a:g}; {b:g}]",
            f"Крок розбиття h:      {h:g}",
            f"Кількість проміжків:  n = (b - a)/h = {n}",
            f"Кількість вузлів:     n + 1 = {n + 1}",
            "",
            "Вузли сітки (x_i = a + i*h):",
        ]
        for i, xv in enumerate(nodes):
            lines.append(f"  x_{i} = {xv:.8g}")

        for key in ("linear", "cubic"):
            info = METHODS[key]
            func, width, tag = info["func"], info["width"], info["tag"]
            ghost = width - 1
            k_min, k_max = -ghost, n - 1
            cnt = k_max - k_min + 1
            lines += [
                "", f"=== {info['name'].upper()} ===",
                f"Фіктивні вузли: {ghost} зліва (x_{-ghost} … x_-1) і {ghost} справа (x_{n + 1} … x_{n + ghost})"
            ]
            lines.append("  зліва:  " + ", ".join(f"x_{i} = {a + i * h:.6g}" for i in range(-ghost, 0)))
            lines.append("  справа: " + ", ".join(f"x_{i} = {a + i * h:.6g}" for i in range(n + 1, n + ghost + 1)))
            lines += [
                f"Кількість сплайнів: {cnt}  (B_{tag},{k_min} … B_{tag},{k_max})",
                f"Носій B_{tag},k: [x_k; x_(k+{width})], довжина {width} проміжки",
                "Частини сплайнів за межами [a; b] показано штрихпунктиром.",
                "",
                f"Значення B_{tag},0 у вузлах носія:",
            ]
            for q in range(width + 1):
                lab = "x_0" if q == 0 else f"x_{q}"
                xq = a + q * h
                val = float(func(xq, 0, a, h))
                lines.append(f"  {lab:<5} = {xq:>10.6g} | B = {val:.6f}")
            lines += ["", "Многочлени на проміжках (t = (x - x_k)/h):"]
            for q, poly in enumerate(info["pieces"]):
                lines.append(f"  [x_k+{q}; x_k+{q + 1}]: B = {poly}")

        lines += ["", "Побудову завершено."]
        self.write_log("\n".join(lines))