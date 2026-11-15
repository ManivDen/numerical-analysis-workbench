import tkinter as tk
from tkinter import ttk

# Чіткий інтерфейс на екранах із масштабуванням (Windows)
try:
    from ctypes import windll
    windll.shcore.SetProcessDpiAwareness(1)
except Exception:
    pass

from ui.widgets import SwitchTab
from ui.slae_tab import SlarPanel
from ui.root_tab import RootPanel
from ui.interp_tab import InterpolationTab
from ui.spline_tab import SplineTab
from ui.lsq_tab import LeastSquaresTab


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Чисельні методи: Лінійні, Нелінійні та Інтерполяція")
        self.geometry("1250x820")

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True)

        self.slar_tab = SwitchTab(
            self.notebook, "Оберіть метод:",
            [("Гаусс", "gauss"), ("Якобі", "jacobi")],
            {"gauss": lambda m, s: SlarPanel(m, False, s),
             "jacobi": lambda m, s: SlarPanel(m, True, s)}
        )
        self.notebook.add(self.slar_tab, text="📄 Лінійні рівняння(СЛАР)")

        self.nonlinear_tab = SwitchTab(
            self.notebook, "Оберіть метод:",
            [("Проста ітерація", "iter"), ("Ньютон", "newton")],
            {"iter": lambda m, s: RootPanel(m, "iter", s),
             "newton": lambda m, s: RootPanel(m, "newton", s)}
        )
        self.notebook.add(self.nonlinear_tab, text="📈 Нелінійні рівняння")

        self.interp_tab = InterpolationTab(self.notebook)
        self.notebook.add(self.interp_tab, text="📊 Інтерполяція(Лагранж/Ньютон)")

        self.spline_tab = SplineTab(self.notebook)
        self.notebook.add(self.spline_tab, text="〰 B-сплайни")

        self.lsq_tab = LeastSquaresTab(self.notebook)
        self.notebook.add(self.lsq_tab, text="📉 НСКН")

        self.notebook.select(self.slar_tab)


if __name__ == "__main__":
    App().mainloop()