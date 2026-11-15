import os
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from core.parser import evaluate_expression
from core.least_squares import fit_models
from ui.widgets import make_log, to_float

class LeastSquaresTab(ttk.Frame):
    MIN_POINTS, MAX_POINTS = 1, 200
    ERRORS_FILE = "errors.txt"

    def __init__(self, master):
        super().__init__(master)
        left = ttk.Frame(self, padding=10)
        left.pack(side="left", fill="y")
        right = ttk.Frame(self)
        right.pack(side="right", fill="both", expand=True)

        ttk.Label(left, text="Апроксимація методом найменших квадратів",
                  font=("Segoe UI", 13, "bold")).pack(anchor="w", pady=(0, 4))
        ttk.Label(left, justify="left",
                  text="Точки x_i = a + i·h беруться з функції f(x) на [A; B] і наближуються:\n"
                       "парабола y = ax² + bx + c, пряма y = ax + b, гіпербола y = a + b/x.").pack(
            anchor="w", pady=(0, 8))

        grid = ttk.Frame(left)
        grid.pack(anchor="w", fill="x")
        self.f_var = tk.StringVar(value="sin(x) + 0.5*x**2")
        self.a_var = tk.StringVar(value="1")
        self.b_var = tk.StringVar(value="6")
        self.n_var = tk.StringVar(value="6")
        fields = [("Функція f(x):", self.f_var, 26),
                  ("Ліва межа [A]:", self.a_var, 14),
                  ("Права межа [B]:", self.b_var, 14),
                  ("Кількість точок [n]:", self.n_var, 14)]
        for r, (text, var, w) in enumerate(fields):
            ttk.Label(grid, text=text).grid(row=r, column=0, sticky="w", pady=3, padx=(0, 8))
            ttk.Entry(grid, textvariable=var, width=w).grid(row=r, column=1, sticky="w", pady=3)

        ttk.Label(
            left, wraplength=400, justify="left", foreground="#555",
            text="Степінь: ** або ^. Доступні sin, cos, tan, exp, ln, log, lg, log10, sqrt, abs, "
                 "а також pi та e. Множення пишіть явно: 2*x. Межі можуть бути від'ємними. "
                 "Модель обчислюється, лише якщо точок достатньо: парабола n ≥ 3, пряма та "
                 "гіпербола n ≥ 2. Гіпербола не визначена, якщо одна з точок x = 0."
        ).pack(anchor="w", pady=(6, 8))

        ttk.Button(left, text="🚀 Побудувати апроксимацію",
                   command=self.calculate).pack(fill="x", ipady=6, pady=(0, 8))

        ttk.Label(left, text="Результати обчислень:").pack(anchor="w")
        log_frame, self.log = make_log(left, width=52, height=18)
        log_frame.pack(fill="both", expand=True)

        self.figure = Figure(figsize=(8, 6), tight_layout=True)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=right)
        self.toolbar = NavigationToolbar2Tk(self.canvas, right)
        self.toolbar.update()
        self.toolbar.pack(side="bottom", fill="x")
        self.canvas.get_tk_widget().pack(side="top", fill="both", expand=True)

    def write_log(self, text):
        self.log.delete("1.0", "end")
        self.log.insert("end", text)

    def read_input(self):
        expression = self.f_var.get().strip()
        if not expression:
            raise ValueError("Введи функцію f(x).")
        A = to_float(self.a_var.get(), "A")
        B = to_float(self.b_var.get(), "B")
        try:
            n = int(self.n_var.get().strip())
        except ValueError:
            raise ValueError("Поле «n» має бути цілим числом.")
        if A >= B:
            raise ValueError("Ліва межа A повинна бути меншою за B.")
        if not self.MIN_POINTS <= n <= self.MAX_POINTS:
            raise ValueError(f"Кількість точок n має бути від {self.MIN_POINTS} до {self.MAX_POINTS}.")
        return expression, A, B, n

    @staticmethod
    def fit_models(x, y):
        """Повертає {ключ: (коефіцієнти, функція)} або {ключ: None} + причини відмови."""
        n = len(x)
        models, reasons = {}, {}

        def solve(matrix, vector):
            try:
                return np.linalg.solve(np.array(matrix, dtype=float), np.array(vector, dtype=float))
            except np.linalg.LinAlgError:
                return None

        # ----- Квадратична модель: y = a*x^2 + b*x + c -----
        coef = None
        if n >= 3:
            coef = solve([[np.sum(x**4), np.sum(x**3), np.sum(x**2)],
                          [np.sum(x**3), np.sum(x**2), np.sum(x)],
                          [np.sum(x**2), np.sum(x), n]],
                         [np.sum(x**2 * y), np.sum(x * y), np.sum(y)])
        if coef is None:
            models["quad"] = None
            reasons["quad"] = "потрібно щонайменше 3 різні точки (СЛАР виродилась)"
        else:
            a, b, c = coef
            models["quad"] = (coef, lambda t, a=a, b=b, c=c: a * t**2 + b * t + c)

        # ----- Лінійна модель: y = a*x + b -----
        coef = None
        if n >= 2:
            coef = solve([[np.sum(x**2), np.sum(x)], [np.sum(x), n]],
                         [np.sum(x * y), np.sum(y)])
        if coef is None:
            models["lin"] = None
            reasons["lin"] = "потрібно щонайменше 2 різні точки (СЛАР виродилась)"
        else:
            a, b = coef
            models["lin"] = (coef, lambda t, a=a, b=b: a * t + b)

        # ----- Гіпербола: y = a + b/x -----
        coef = None
        scale = max(np.max(np.abs(x)), 1.0)
        if np.any(np.abs(x) < 1e-9 * scale):
            reasons["hyp"] = "одна з точок має x = 0, а 1/x там не існує"
        elif n < 2:
            reasons["hyp"] = "потрібно щонайменше 2 різні точки (СЛАР виродилась)"
        else:
            coef = solve([[n, np.sum(1 / x)], [np.sum(1 / x), np.sum(1 / x**2)]],
                         [np.sum(y), np.sum(y / x)])
            if coef is None:
                reasons["hyp"] = "СЛАР виродилась"
        if coef is None:
            models["hyp"] = None
        else:
            a, b = coef
            models["hyp"] = (coef, lambda t, a=a, b=b: a + b / t)
        return models, reasons

    def calculate(self):
        try:
            expression, A, B, n = self.read_input()

            x = np.linspace(A, B, n)
            with np.errstate(all="ignore"):
                y = np.asarray(evaluate_expression(expression, x), dtype=float)
            if y.ndim == 0:
                y = np.full_like(x, float(y))
            if y.shape != x.shape:
                raise ValueError("Функція повинна повертати значення для всіх x.")
            if not np.all(np.isfinite(y)):
                raise ValueError("Функція має невизначені значення на проміжку.")

            models, reasons = self.fit_models(x, y)
            names = {"quad": "Квадратична", "lin": "Лінійна", "hyp": "Гіперболічна"}
            forms = {"quad": "y = a*x^2 + b*x + c", "lin": "y = a*x + b", "hyp": "y = a + b/x"}
            errors, sums = {}, {}
            for key, model in models.items():
                if model is not None:
                    errors[key] = np.abs(y - model[1](x))
                    sums[key] = float(errors[key].sum())

            def coef_text(key, digits):
                c = models[key][0]
                letters = "abc"
                return ", ".join(f"{letters[i]} = {v:.{digits}f}" for i, v in enumerate(c))

            def cell(key, i, width):
                return f"{errors[key][i]:>{width}.6f}" if key in errors else f"{'—':>{width}}"

            # ----- Запис у файл -----
            with open(self.ERRORS_FILE, "w", encoding="utf-8") as file:
                file.write("Результати апроксимації методом найменших квадратів\n\n")
                file.write(f"Функція: f(x) = {expression}\n")
                file.write(f"Проміжок: [{A}; {B}], кількість точок: {n}\n\n")
                for key in ("quad", "lin", "hyp"):
                    file.write(f"{names[key]} модель {forms[key]}\n")
                    if key in errors:
                        file.write(f"{coef_text(key, 8)}\n")
                        file.write(f"Сумарна абсолютна похибка = {sums[key]:.8f}\n\n")
                    else:
                        file.write(f"не обчислено: {reasons[key]}\n\n")
                file.write("Похибки в точках:\n")
                file.write(f"{'x':>12} {'f(x)':>14} {'Квадратична':>14} {'Лінійна':>14} {'Гіпербола':>14}\n")
                for i in range(n):
                    file.write(f"{x[i]:>12.6f} {y[i]:>14.6f} {cell('quad', i, 14)} "
                               f"{cell('lin', i, 14)} {cell('hyp', i, 14)}\n")

            # ----- Протокол у вікні -----
            lines = [f"Функція: f(x) = {expression}",
                     f"Проміжок: [{A:g}; {B:g}]",
                     f"Кількість точок: {n}", ""]
            if "quad" in errors:
                a, b, c = models["quad"][0]
                lines += ["КВАДРАТИЧНА МОДЕЛЬ", f"y = {a:.6f}x² + ({b:.6f})x + ({c:.6f})",
                          f"Сумарна похибка: {sums['quad']:.6f}", ""]
            if "lin" in errors:
                a, b = models["lin"][0]
                lines += ["ЛІНІЙНА МОДЕЛЬ", f"y = {a:.6f}x + ({b:.6f})",
                          f"Сумарна похибка: {sums['lin']:.6f}", ""]
            if "hyp" in errors:
                a, b = models["hyp"][0]
                lines += ["ГІПЕРБОЛІЧНА МОДЕЛЬ", f"y = {a:.6f} + ({b:.6f})/x",
                          f"Сумарна похибка: {sums['hyp']:.6f}", ""]
            skipped = [key for key in ("quad", "lin", "hyp") if key not in errors]
            if skipped:
                lines.append("НЕ ОБЧИСЛЕНО")
                lines += [f"{names[key]}: {reasons[key]}" for key in skipped]
                lines.append("")
            if n == 3 and "quad" in errors:
                lines += ["Примітка: через 3 точки парабола проходить точно, тому її похибка ≈ 0.", ""]
            if n == 2:
                lines += ["Примітка: через 2 точки пряма та гіпербола проходять точно, "
                          "тому їхня похибка ≈ 0.", ""]

            if errors:
                lines += ["ПОХИБКИ В ТОЧКАХ |f(x_i) − y(x_i)|",
                          f"{'x':>10} {'f(x)':>12} {'Квадр.':>10} {'Лін.':>10} {'Гіпербола':>10}"]
                for i in range(n):
                    lines.append(f"{x[i]:>10.4f} {y[i]:>12.6f} {cell('quad', i, 10)} "
                                 f"{cell('lin', i, 10)} {cell('hyp', i, 10)}")
                best = min(sums, key=sums.get)
                lines += ["", f"Найкраща за сумою абсолютних похибок: {names[best]}"]
            else:
                lines.append("З такою кількістю точок жодну модель побудувати неможливо.")
            lines.append(f"Результати записано у файл {os.path.abspath(self.ERRORS_FILE)}")
            self.write_log("\n".join(lines))

            # ----- Графік -----
            ax = self.ax
            ax.clear()
            x_plot = np.linspace(A, B, 400)
            ax.scatter(x, y, color="black", s=55, label="Початкові точки", zorder=5)
            if "quad" in models and models["quad"] is not None:
                ax.plot(x_plot, models["quad"][1](x_plot), lw=2, label="Квадратична апроксимація")
            if "lin" in models and models["lin"] is not None:
                ax.plot(x_plot, models["lin"][1](x_plot), lw=2, label="Лінійна апроксимація")
            if models["hyp"] is not None:
                hyp = models["hyp"][1]
                pole_inside = A < 0 < B
                color = "tab:green"
                # гілки малюємо окремо, щоб крива не з'єднувалась через розрив у x = 0
                for k, mask in enumerate((x_plot < 0, x_plot > 0)):
                    if mask.sum() > 1:
                        ax.plot(x_plot[mask], hyp(x_plot[mask]), lw=2, color=color,
                                label="Гіперболічна апроксимація" if k == 0 or not (x_plot < 0).sum() > 1 else None)
                if pole_inside:
                    ax.axvline(0, color="gray", ls="--", lw=1)
                    # обмежуємо вісь y, щоб полюс гіперболи не «розтягував» графік
                    ref = [y]
                    for key in ("quad", "lin"):
                        if models[key] is not None:
                            ref.append(models[key][1](x_plot))
                    ref = np.concatenate(ref)
                    lo, hi = float(ref.min()), float(ref.max())
                    pad = 0.5 * (hi - lo if hi > lo else max(abs(hi), 1.0))
                    ax.set_ylim(lo - pad, hi + pad)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_title("Апроксимація методом найменших квадратів")
            ax.grid(True, ls=":", alpha=0.7)
            ax.legend()
            self.figure.tight_layout()
            self.canvas.draw()

        except (ValueError, SyntaxError, OverflowError, ZeroDivisionError, TypeError) as error:
            messagebox.showwarning("Перевір вхідні дані", str(error))
        except OSError as error:
            messagebox.showerror("Помилка запису файлу", str(error))
        except Exception as error:
            messagebox.showerror("Помилка обчислення", f"Сталася помилка:\n{error}")