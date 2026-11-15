import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np

from core.parser import evaluate_expression
from core.interpolation import (
    lagrange_interpolation,
    newton_forward_interpolation,
    newton_divided_differences,
    newton_divided_interpolation,
    chebyshev_roots,
)
from ui.widgets import make_log, to_float, PlotTab

class InterpolationTab(ttk.Frame):
    MAX_INTERVALS = 30

    def __init__(self, master):
        super().__init__(master)
        left = ttk.Frame(self, padding=10)
        left.pack(side="left", fill="y")
        right = ttk.Frame(self)
        right.pack(side="right", fill="both", expand=True)

        ttk.Label(left, text="Інтерполяція Лагранжа і Ньютона",
                  font=("Segoe UI", 13, "bold")).pack(anchor="w", pady=(0, 6))

        grid = ttk.Frame(left)
        grid.pack(anchor="w", fill="x")
        self.function_var = tk.StringVar(value="lg(x)")
        self.a_var = tk.StringVar(value="1000")
        self.b_var = tk.StringVar(value="1050")
        self.h_var = tk.StringVar(value="10")
        fields = [("Функція f(x):", self.function_var, 22),
                  ("Ліва межа [a = x₀]:", self.a_var, 14),
                  ("Права межа [b]:", self.b_var, 14),
                  ("Крок вузлів [h]:", self.h_var, 14)]
        for r, (text, var, w) in enumerate(fields):
            ttk.Label(grid, text=text).grid(row=r, column=0, sticky="w", pady=3, padx=(0, 8))
            ttk.Entry(grid, textvariable=var, width=w).grid(row=r, column=1, sticky="w", pady=3)

        ttk.Label(
            left, wraplength=400, justify="left", foreground="#555",
            text="Степінь: ** або ^. Доступні sin, cos, tan, exp, ln, log, lg, sqrt, abs, erf, Phi, "
                 "а також pi та e. Множення пишіть явно: 2*x. Вузли x_i = a + i·h, крок h має "
                 "точно ділити b−a. Для ln(x) та lg(x) потрібно x > 0 на всьому проміжку."
        ).pack(anchor="w", pady=(6, 4))

        self.nodes_preview = ttk.Label(left, wraplength=400, justify="left")
        self.nodes_preview.pack(anchor="w", pady=(0, 6))
        for var in (self.a_var, self.b_var, self.h_var):
            var.trace_add("write", lambda *_: self.update_nodes_preview())
        self.update_nodes_preview()

        ttk.Button(left, text="🚀 Обчислити всі три завдання",
                   command=self.calculate_all).pack(fill="x", ipady=6, pady=(0, 8))

        ttk.Label(left, text="Протокол обчислень:").pack(anchor="w")
        log_frame, self.report = make_log(left, width=52, height=14)
        log_frame.pack(fill="both", expand=True)

        self.tabs = ttk.Notebook(right)
        self.task1_tab = PlotTab(self.tabs)
        self.task2_tab = PlotTab(self.tabs)
        self.task3_tab = PlotTab(self.tabs)
        self.tabs.add(self.task1_tab, text="Завдання 1 — рівновіддалені вузли")
        self.tabs.add(self.task2_tab, text="Завдання 2 — вузли Чебишова")
        self.tabs.add(self.task3_tab, text="Завдання 3 — похибки")
        self.tabs.pack(fill="both", expand=True)

    def read_numbers(self):
        a = to_float(self.a_var.get(), "a")
        b = to_float(self.b_var.get(), "b")
        h = to_float(self.h_var.get(), "h")
        return a, b, h

    def update_nodes_preview(self):
        try:
            a, b, h = self.read_numbers()
        except ValueError:
            self.nodes_preview.config(text="Межі a, b та крок h мають бути числами.")
            return
        if b <= a or h <= 0:
            self.nodes_preview.config(text="Перевір межі: має бути b > a та h > 0.")
            return
        ratio = (b - a) / h
        intervals = round(ratio)
        if intervals < 1 or not np.isclose(ratio, intervals, rtol=1e-9, atol=1e-8):
            self.nodes_preview.config(
                text=f"Для цих значень (b−a)/h = {ratio:.8g}, тобто крок h не ділить проміжок "
                     "на цілу кількість частин. Зміни h або межі.")
            return
        if intervals > self.MAX_INTERVALS:
            self.nodes_preview.config(
                text=f"Виходить {intervals} проміжків. Для стабільного обчислення обери h так, "
                     f"щоб було не більше {self.MAX_INTERVALS} проміжків.")
            return
        nodes = a + np.arange(intervals + 1) * h
        nodes[-1] = b
        shown = ", ".join(f"{x:.8g}" for x in nodes)
        self.nodes_preview.config(
            text=f"Вузли першого завдання: {shown}\n"
                 f"Проміжків: {intervals}; вузлів: {intervals + 1}; степінь полінома: n={intervals}.")

    def write_report(self, text):
        self.report.delete("1.0", "end")
        self.report.insert("end", text)

    def calculate_all(self):
        function_name = self.function_var.get().strip()
        try:
            if not function_name:
                raise ValueError("Введи формулу функції, наприклад x**3 + 2*x - 56.")
            function = lambda x: evaluate_expression(function_name, x)
            a, b, h = self.read_numbers()
        except (ValueError, SyntaxError) as error:
            messagebox.showwarning("Помилка у формулі", str(error))
            return

        try:
            if not a < b:
                raise ValueError("Права межа b має бути більшою за ліву межу a.")
            if h <= 0:
                raise ValueError("Крок h має бути більшим за нуль.")
            ratio = (b - a) / h
            intervals = int(round(ratio))
            if not np.isclose(ratio, intervals, rtol=1e-9, atol=1e-8):
                raise ValueError(
                    f"Крок h={h:g} не підходить до проміжку [{a:g}; {b:g}]. "
                    f"(b−a)/h = {ratio:.10g}, а має бути цілим числом. "
                    "Наприклад, для [1000;1050] можна взяти h=10, 5 або 2."
                )
            if intervals < 1:
                raise ValueError("Крок h завеликий: на проміжку має бути хоча б два вузли.")
            if intervals > self.MAX_INTERVALS:
                raise ValueError(
                    f"Отримано {intervals} проміжків. Для цієї навчальної програми обери h так, "
                    f"щоб кількість проміжків не перевищувала {self.MAX_INTERVALS}."
                )

            node_count = intervals + 1
            x_equal = a + np.arange(node_count, dtype=float) * h
            x_equal[-1] = b

            x_plot = np.linspace(a, b, 1600)
            with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
                y_true_plot = np.asarray(function(x_plot), dtype=float)
                y_equal = np.asarray(function(x_equal), dtype=float)

            if not np.all(np.isfinite(y_true_plot)) or not np.all(np.isfinite(y_equal)):
                raise ValueError(
                    f"Функція {function_name} дає нескінченні або нечислові значення на цьому проміжку. "
                    "Перевір формулу та область визначення функції. Для exp(x) на великих додатних x "
                    "можливе переповнення чисел."
                )

            equal_lagrange = lagrange_interpolation(x_equal, y_equal, x_plot)
            equal_newton = newton_forward_interpolation(x_equal, y_equal, h, x_plot)

            x_cheb = chebyshev_roots(a, b, node_count)
            with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
                y_cheb = np.asarray(function(x_cheb), dtype=float)
            if not np.all(np.isfinite(y_cheb)):
                raise ValueError("Функція має нечислові значення у вузлах Чебишова. Перевір формулу та область визначення.")
            cheb_lagrange = lagrange_interpolation(x_cheb, y_cheb, x_plot)
            cheb_coefficients = newton_divided_differences(x_cheb, y_cheb)
            cheb_newton = newton_divided_interpolation(x_cheb, cheb_coefficients, x_plot)

            error_equal_lagrange = np.abs(equal_lagrange - y_true_plot)
            error_equal_newton = np.abs(equal_newton - y_true_plot)
            error_cheb_lagrange = np.abs(cheb_lagrange - y_true_plot)
            error_cheb_newton = np.abs(cheb_newton - y_true_plot)
            all_errors = [error_equal_lagrange, error_equal_newton, error_cheb_lagrange, error_cheb_newton]
            if not all(np.all(np.isfinite(err)) for err in all_errors):
                raise ValueError(
                    "Під час обчислення полінома виникло переповнення чисел. "
                    "Спробуй менший степінь (більший h) або перевір проміжок і функцію."
                )

            max_equal_lagrange = float(np.max(error_equal_lagrange))
            max_equal_newton = float(np.max(error_equal_newton))
            max_cheb_lagrange = float(np.max(error_cheb_lagrange))
            max_cheb_newton = float(np.max(error_cheb_newton))

            def plot_task1(ax):
                ax.plot(x_plot, y_true_plot, label="Початкова функція f(x)", linewidth=2.2)
                ax.plot(x_plot, equal_lagrange, "--", label="Поліном Лагранжа")
                ax.plot(x_plot, equal_newton, ":", label="Поліном Ньютона вперед", linewidth=2.2)
                ax.scatter(x_equal, y_equal, color="black", zorder=4, label="Рівновіддалені вузли")
                ax.set_title("Завдання 1. Рівновіддалені вузли")
                ax.set_xlabel("x")
                ax.set_ylabel("y")
                ax.legend()

            def plot_task2(ax):
                ax.plot(x_plot, y_true_plot, label="Початкова функція f(x)", linewidth=2.2)
                ax.plot(x_plot, cheb_lagrange, "--", label="Поліном Лагранжа")
                ax.plot(x_plot, cheb_newton, ":", label="Поліном Ньютона (розділені різниці)", linewidth=2.2)
                ax.scatter(x_cheb, y_cheb, color="black", zorder=4, label="Вузли Чебишова")
                ax.set_title("Завдання 2. Корені полінома Чебишова")
                ax.set_xlabel("x")
                ax.set_ylabel("y")
                ax.legend()

            def plot_task3(ax):
                ax.plot(x_plot, error_equal_lagrange, label="Лагранж — рівновіддалені вузли")
                ax.plot(x_plot, error_equal_newton, "--", label="Ньютон — рівновіддалені вузли")
                ax.plot(x_plot, error_cheb_lagrange, label="Лагранж — вузли Чебишова")
                ax.plot(x_plot, error_cheb_newton, "--", label="Ньютон — вузли Чебишова")
                ax.set_title("Завдання 3. Абсолютна похибка ε(x)=|Pₙ(x)−f(x)|")
                ax.set_xlabel("x")
                ax.set_ylabel("Абсолютна похибка")
                ax.legend(fontsize=8)

            self.task1_tab.draw_plot(plot_task1)
            self.task2_tab.draw_plot(plot_task2)
            self.task3_tab.draw_plot(plot_task3)

            equal_max = max(max_equal_lagrange, max_equal_newton)
            cheb_max = max(max_cheb_lagrange, max_cheb_newton)
            if cheb_max < equal_max and not np.isclose(cheb_max, equal_max, rtol=1e-7, atol=1e-12):
                comparison = "Висновок: у цьому числовому експерименті максимальна похибка на вузлах Чебишова менша."
            elif np.isclose(cheb_max, equal_max, rtol=1e-7, atol=1e-12):
                comparison = "Висновок: максимальні похибки для обох наборів вузлів практично однакові."
            else:
                comparison = (
                    "Висновок: на цьому проміжку похибка з вузлами Чебишова не виявилася меншою. "
                    "Це залежить від функції та степеня полінома; висновок зроблено за обчисленнями."
                )

            def format_nodes(title, nodes, values):
                lines = [title, " i              x_i                 y_i=f(x_i)"]
                for i, (xx, yy) in enumerate(zip(nodes, values)):
                    lines.append(f" {i:<3} {xx:>18.10g} {yy:>24.12g}")
                return "\n".join(lines)

            report = [
                f"Функція: {function_name}",
                f"Проміжок: [{a:g}; {b:g}]",
                f"Крок рівновіддалених вузлів h = {h:g}",
                f"Кількість проміжків n = {intervals}; кількість вузлів = n+1 = {node_count}",
                "",
                format_nodes("ЗАВДАННЯ 1. Рівновіддалені вузли:", x_equal, y_equal),
                "",
                format_nodes("ЗАВДАННЯ 2. Вузли Чебишова:", x_cheb, y_cheb),
                "",
                "ЗАВДАННЯ 3. Максимальна абсолютна похибка на сітці з 1600 точок:",
                f"Лагранж, рівновіддалені вузли: {max_equal_lagrange:.12g}",
                f"Ньютон, рівновіддалені вузли:  {max_equal_newton:.12g}",
                f"Лагранж, вузли Чебишева:       {max_cheb_lagrange:.12g}",
                f"Ньютон, вузли Чебишева:        {max_cheb_newton:.12g}",
                "",
                comparison,
                "",
                "Пояснення: n — степінь полінома; n+1 — кількість вузлів. У першому завданні",
                "Ньютон обчислюється за формулою вперед зі скінченними різницями та кроком h.",
                "Для вузлів Чебишова, які не є рівновіддаленими, використано формулу Ньютона",
                "з розділеними різницями. Максимум похибки оцінено на 1600 точках.",
            ]
            self.write_report("\n".join(report))
            self.tabs.select(0)

        except (ValueError, SyntaxError, FloatingPointError, OverflowError, ZeroDivisionError, TypeError) as error:
            messagebox.showwarning("Перевір вхідні дані", str(error))
        except Exception as error:
            messagebox.showerror("Помилка обчислення", f"Сталася помилка:\n{error}")