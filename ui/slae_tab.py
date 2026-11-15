import contextlib
import tkinter as tk
from tkinter import ttk

from core.slae import solve_gaussian, solve_jacobi
from ui.widgets import make_log, TextRedirect

class SlarPanel(ttk.Frame):
    MIN_N, MAX_N = 2, 10

    def __init__(self, master, is_jacobi, selector):
        super().__init__(master)
        self.is_jacobi = is_jacobi
        self.cells_a = {}
        self.cells_b = {}

        left = ttk.Frame(self, padding=10)
        left.pack(side="left", fill="y")
        right = ttk.Frame(self, padding=(0, 10, 10, 10))
        right.pack(side="right", fill="both", expand=True)

        selector(left).pack(anchor="w")

        ttk.Label(left, text="Система A·x = b" +
                  ("\nІтераційний метод Якобі" if is_jacobi else "\nПрямий хід + зворотний хід"),
                  justify="left").pack(anchor="w", pady=(0, 8))

        params = ttk.Frame(left)
        params.pack(anchor="w", fill="x")
        self.n_var = tk.StringVar(value="3")
        ttk.Label(params, text="Розмірність [n]:").grid(row=0, column=0, sticky="w", pady=3, padx=(0, 8))
        spin = ttk.Spinbox(params, from_=self.MIN_N, to=self.MAX_N, textvariable=self.n_var,
                           width=12, command=self.resize)
        spin.grid(row=0, column=1, sticky="w", pady=3)
        spin.bind("<Return>", lambda e: self.resize())
        spin.bind("<FocusOut>", lambda e: self.resize())

        self.eps_var = tk.StringVar(value="0.001")
        if is_jacobi:
            ttk.Label(params, text="Точність [ε]:").grid(row=1, column=0, sticky="w", pady=3, padx=(0, 8))
            ttk.Entry(params, textvariable=self.eps_var, width=14).grid(row=1, column=1, sticky="w", pady=3)

        ttk.Label(left, text="Матриця A та вектор b (порожня клітинка = 0):").pack(anchor="w", pady=(8, 2))
        self.matrix_frame = ttk.Frame(left)
        self.matrix_frame.pack(anchor="w")

        ttk.Button(left, text="🚀 Розрахувати", command=self.do_calc).pack(fill="x", ipady=6, pady=(12, 0))

        ttk.Label(right, text="Протокол обчислень:").pack(anchor="w")
        log_frame, self.log = make_log(right)
        log_frame.pack(fill="both", expand=True)

        self.build_matrix(3)

    def current_n(self):
        try:
            n = int(self.n_var.get().strip())
        except ValueError:
            return None
        return max(self.MIN_N, min(self.MAX_N, n))

    def resize(self):
        n = self.current_n()
        if n is None:
            return
        self.n_var.set(str(n))
        self.build_matrix(n)

    def build_matrix(self, n):
        for w in self.matrix_frame.winfo_children():
            w.destroy()
        for j in range(n):
            ttk.Label(self.matrix_frame, text=f"x{j + 1}").grid(row=0, column=j, pady=(0, 2))
        ttk.Label(self.matrix_frame, text="b").grid(row=0, column=n + 1, pady=(0, 2))
        for i in range(n):
            for j in range(n):
                var = self.cells_a.setdefault((i, j), tk.StringVar())
                ttk.Entry(self.matrix_frame, textvariable=var, width=7,
                          justify="center").grid(row=i + 1, column=j, padx=1, pady=1)
            ttk.Label(self.matrix_frame, text="|").grid(row=i + 1, column=n, padx=2)
            var = self.cells_b.setdefault(i, tk.StringVar())
            ttk.Entry(self.matrix_frame, textvariable=var, width=7,
                      justify="center").grid(row=i + 1, column=n + 1, padx=1, pady=1)

    def do_calc(self):
        self.log.delete("1.0", "end")
        with contextlib.redirect_stdout(TextRedirect(self.log)):
            try:
                n = self.current_n()
                if n is None:
                    raise ValueError("Розмірність n має бути цілим числом.")
                self.build_matrix(n)

                def val(var):
                    text = var.get().strip()
                    return float(text) if text else 0.0

                a = [[val(self.cells_a.setdefault((i, j), tk.StringVar())) for j in range(n)]
                     for i in range(n)]
                b = [val(self.cells_b.setdefault(i, tk.StringVar())) for i in range(n)]

                if self.is_jacobi:
                    res = solve_jacobi(a, b, float(self.eps_var.get()))
                else:
                    res = solve_gaussian(a, b)

                print("\nфінальний результат:")
                for idx, v in enumerate(res):
                    print(f"x{idx + 1} = {v:.4f}")
            except Exception as e:
                print(f"\nпомилка: {e}")