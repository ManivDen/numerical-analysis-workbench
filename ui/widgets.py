import tkinter as tk
from tkinter import ttk, font as tkfont
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

def pick_mono_font():
    """Моноширинний шрифт з нормальною кирилицею."""
    available = set(tkfont.families())
    for name in ("Consolas", "Menlo", "DejaVu Sans Mono", "Courier New", "Courier"):
        if name in available:
            return name
    return "TkFixedFont"

def method_selector(parent, title, var, options, command):
    """Блок «Оберіть метод» у стилі вкладки B-сплайнів."""
    box = ttk.Frame(parent)
    ttk.Label(box, text=title, font=("Segoe UI", 13, "bold")).pack(anchor="w", pady=(0, 4))
    row = ttk.Frame(box)
    row.pack(anchor="w", pady=(0, 8))
    for text, value in options:
        ttk.Radiobutton(row, text=text, variable=var, value=value,
                        command=command).pack(side="left", padx=(0, 12))
    return box


def make_log(parent, width=52, height=20):
    """Темне текстове поле «Протокол» з прокруткою."""
    frame = ttk.Frame(parent)
    text = tk.Text(frame, width=width, height=height, bg="#111111", fg="#f0f0f0",
                   insertbackground="white", font=(pick_mono_font(), 11), wrap="none")
    vs = ttk.Scrollbar(frame, command=text.yview)
    hs = ttk.Scrollbar(frame, orient="horizontal", command=text.xview)
    text.configure(yscrollcommand=vs.set, xscrollcommand=hs.set)
    hs.pack(side="bottom", fill="x")
    vs.pack(side="right", fill="y")
    text.pack(side="left", fill="both", expand=True)
    return frame, text


class SwitchTab(ttk.Frame):
    """Вкладка з перемикачем методів (радіокнопки зліва, як у B-сплайнах)."""

    def __init__(self, master, title, options, factories):
        super().__init__(master)
        self.var = tk.StringVar(value=options[0][1])
        self.panels = {}

        def selector(parent):
            return method_selector(parent, title, self.var, options, self.switch)

        for _, key in options:
            self.panels[key] = factories[key](self, selector)
        self.switch()

    def switch(self):
        for panel in self.panels.values():
            panel.pack_forget()
        self.panels[self.var.get()].pack(fill="both", expand=True)


class TextRedirect:
    """Перехоплення print() у текстове поле."""

    def __init__(self, widget):
        self.widget = widget

    def write(self, text):
        self.widget.insert("end", text)
        self.widget.see("end")

    def flush(self):
        pass


def to_float(s, name):
    try:
        return float(s.strip().replace(",", "."))
    except ValueError:
        raise ValueError(f"Поле «{name}» має бути числом.")

class PlotTab(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.figure = Figure(figsize=(8, 5), tight_layout=True)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self)
        self.toolbar.update()
        self.toolbar.pack(side="bottom", fill="x")
        self.canvas.get_tk_widget().pack(side="top", fill="both", expand=True)

    def draw_plot(self, draw_function):
        self.figure.clear()
        axes = self.figure.add_subplot(111)
        draw_function(axes)
        axes.grid(True, alpha=0.3)
        self.figure.tight_layout()
        self.canvas.draw()