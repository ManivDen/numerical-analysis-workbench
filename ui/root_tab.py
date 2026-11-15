import os
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import pandas as pd
import sympy as sp
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

class RootPanel(ttk.Frame):
    DEFAULTS = {
        "iter": dict(title="Метод простої ітерації", label="Функція φ(x):",
                     f="(x + 5)**(1/3)", x0="1.5, 2.0"),
        "newton": dict(title="Метод Ньютона", label="Функція f(x):",
                       f="x**3 - 2*x - 5", x0="2.5, 3.0"),
    }

    def __init__(self, master, mode, selector):
        super().__init__(master)
        self.mode = mode
        self.current_data = {}
        d = self.DEFAULTS[mode]

        left = ttk.Frame(self, padding=10)
        left.pack(side="left", fill="y")
        right = ttk.Frame(self)
        right.pack(side="right", fill="both", expand=True)

        selector(left).pack(anchor="w")
        ttk.Label(left, text="Кілька ε та x₀ можна вводити через кому —\n"
                             "будуть обчислені всі комбінації.",
                  justify="left").pack(anchor="w", pady=(0, 8))

        grid = ttk.Frame(left)
        grid.pack(anchor="w", fill="x")
        self.f_var = tk.StringVar(value=d["f"])
        self.a_var = tk.StringVar(value="1")
        self.b_var = tk.StringVar(value="3")
        self.eps_var = tk.StringVar(value="0.01, 0.001")
        self.x0_var = tk.StringVar(value=d["x0"])
        fields = [(d["label"], self.f_var, 26),
                  ("Ліва межа [a]:", self.a_var, 14),
                  ("Права межа [b]:", self.b_var, 14),
                  ("Точність [ε] через кому:", self.eps_var, 26),
                  ("Поч. набл. [x₀] через кому:", self.x0_var, 26)]
        for r, (text, var, w) in enumerate(fields):
            ttk.Label(grid, text=text).grid(row=r, column=0, sticky="w", pady=3, padx=(0, 8))
            ttk.Entry(grid, textvariable=var, width=w).grid(row=r, column=1, sticky="w", pady=3)

        ttk.Button(left, text="🚀 Розрахувати та побудувати",
                   command=self.calc).pack(fill="x", ipady=6, pady=(12, 6))
        self.export_btn = ttk.Button(left, text="📥 Експорт в Excel",
                                     command=self.export_excel, state="disabled")
        self.export_btn.pack(fill="x", ipady=6, pady=(0, 10))

        self.res_label = ttk.Label(left, text="Очікування даних...", wraplength=380,
                                   justify="left", font=("Segoe UI", 11, "bold"))
        self.res_label.pack(anchor="w")

        self.figure = Figure(figsize=(8, 6), tight_layout=True)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=right)
        self.toolbar = NavigationToolbar2Tk(self.canvas, right)
        self.toolbar.update()
        self.toolbar.pack(side="bottom", fill="x")
        self.canvas.get_tk_widget().pack(side="top", fill="both", expand=True)

    def calc(self):
        try:
            f_str = self.f_var.get().replace("^", "**")
            a, b = float(self.a_var.get()), float(self.b_var.get())
            eps_list = [float(e.strip()) for e in self.eps_var.get().split(",")]
            x0_list = [float(x.strip()) for x in self.x0_var.get().split(",")]

            x = sp.Symbol("x")
            expr = sp.sympify(f_str)
            self.current_data = {"Точність (ε)": [], "Поч. набл. (X0)": [],
                                 "К-ть ітер. (n)": [], "Корінь (Xn)": [], "f(Xn)": []}
            last_x_n, last_iterations = None, []

            if self.mode == "iter":
                phi = sp.lambdify(x, expr, "numpy")
                for eps in eps_list:
                    for x0 in x0_list:
                        x_n = x0
                        iterations = [x_n]
                        for _ in range(100):
                            x_next = phi(x_n)
                            iterations.append(x_next)
                            if abs(x_next - x_n) < eps:
                                x_n = x_next
                                break
                            x_n = x_next
                        f_xn = x_n - phi(x_n)
                        self._store(eps, x0, len(iterations) - 1, x_n, f_xn)
                        last_x_n, last_iterations = x_n, iterations
                func = phi
            else:
                f = sp.lambdify(x, expr, "numpy")
                df = sp.lambdify(x, sp.diff(expr, x), "numpy")
                for eps in eps_list:
                    for x0 in x0_list:
                        x_n = x0
                        iterations = []
                        for _ in range(100):
                            fx_n = f(x_n)
                            dfx_n = df(x_n)
                            if abs(dfx_n) < 1e-10:
                                raise ValueError("Похідна дорівнює нулю — метод Ньютона не застосовний.")
                            x_next = x_n - fx_n / dfx_n
                            iterations.append((x_n, fx_n, x_next))
                            if abs(x_next - x_n) < eps:
                                x_n = x_next
                                break
                            x_n = x_next
                        self._store(eps, x0, len(iterations), x_n, f(x_n))
                        last_x_n, last_iterations = x_n, iterations
                func = f

            self.export_btn.config(state="normal")
            self.res_label.config(
                text=f"Обчислено {len(eps_list) * len(x0_list)} комбінацій.\n"
                     f"Візуал для ост. x0: {last_x_n:.6f}")

            ax = self.ax
            ax.clear()
            x_vals = np.linspace(a, b, 400)
            if self.mode == "iter":
                ax.plot(x_vals, func(x_vals), label=f"φ(x) = {f_str}", color="tab:blue", lw=2)
                ax.plot(x_vals, x_vals, label="y = x", color="black", ls="--", lw=1.5)
                for i in range(len(last_iterations) - 1):
                    xc, xn = last_iterations[i], last_iterations[i + 1]
                    ax.plot([xc, xc], [xc, xn], color="tab:orange", alpha=0.7)
                    ax.plot([xc, xn], [xn, xn], color="tab:orange", alpha=0.7)
                ax.plot(last_x_n, last_x_n, "o", color="red", mec="black", ms=9, label="корінь")
                ax.set_title("Метод простої ітерації")
            else:
                ax.plot(x_vals, func(x_vals), label=f"f(x) = {f_str}", color="tab:blue", lw=2)
                ax.axhline(0, color="black", lw=1.5)
                for xi, fxi, xnext in last_iterations:
                    ax.plot([xi, xi], [0, fxi], color="gray", alpha=0.6)
                    ax.plot([xi, xnext], [fxi, 0], color="tab:orange", alpha=0.8)
                ax.plot(last_x_n, func(last_x_n), "o", color="red", mec="black", ms=9, label="корінь")
                ax.set_xlim([a, b])
                ax.set_title("Метод Ньютона")
            ax.set_xlabel("$x$")
            ax.set_ylabel("$y$")
            ax.legend()
            ax.grid(True, ls=":", alpha=0.7)
            self.figure.tight_layout()
            self.canvas.draw()
        except Exception as e:
            messagebox.showerror("Помилка", str(e))

    def _store(self, eps, x0, n_iter, x_n, f_xn):
        d = self.current_data
        d["Точність (ε)"].append(eps)
        d["Поч. набл. (X0)"].append(x0)
        d["К-ть ітер. (n)"].append(n_iter)
        d["Корінь (Xn)"].append(x_n)
        d["f(Xn)"].append(f_xn)

    def export_excel(self):
        file_path = filedialog.asksaveasfilename(
            title="Зберегти Excel", initialfile="results.xlsx", defaultextension=".xlsx",
            filetypes=[("Excel Files", "*.xlsx")])
        if not file_path:
            return
        try:
            df = pd.DataFrame(self.current_data)
            if os.path.exists(file_path):
                existing_df = pd.read_excel(file_path)
                df = pd.concat([existing_df, df], ignore_index=True)
                # чистимо таблицю від дублікатів, залишаючи найсвіжіші дані
                df.drop_duplicates(subset=["Точність (ε)", "Поч. набл. (X0)"], keep="last", inplace=True)
            df.to_excel(file_path, index=False)
            messagebox.showinfo("Успіх", "Усі комбінації збережено, дублікати видалено.")
        except Exception as e:
            messagebox.showerror("Помилка", str(e))