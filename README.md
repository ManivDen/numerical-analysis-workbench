# Numerical Analysis Workbench

A desktop GUI application written in Python for solving and visualizing classical numerical analysis problems. Built with a clean separation between mathematical solvers (`core/`) and the `tkinter`/`matplotlib` graphical interface (`ui/`).

## Features

* **Systems of Linear Algebraic Equations (SLAE):**
  * **Gaussian Elimination:** Direct solver with forward elimination and back-substitution logging.
  * **Jacobi Method:** Iterative solver with strict diagonal dominance verification and error tracking.
* **Nonlinear Equations:**
  * **Fixed-Point Iteration & Newton's Method:** Symbolic differentiation via `sympy`, cobweb/tangent step visualization, multi-parameter batch evaluation, and `.xlsx` export with duplicate filtering.
* **Polynomial Interpolation:**
  * **Lagrange & Newton Polynomials:** Forward finite differences for equidistant nodes and divided differences for **Chebyshev nodes**, complete with absolute error distribution plots.
* **B-Splines:**
  * Interactive construction and visualization of linear ($m=2$) and cubic ($m=4$) B-splines with ghost node extensions.
* **Least Squares Approximation:**
  * Linear, quadratic, and hyperbolic curve fitting with pole handling and residual analysis.
* **Safe Expression Parser:**
  * Custom mathematical expression evaluator built on Python's `ast` module (Abstract Syntax Tree) instead of unsafe `eval()`.

## Tech Stack

* **Language:** Python 3.10+
* **Math & Data:** NumPy, SciPy, SymPy, Pandas
* **GUI & Visualization:** Tkinter (`ttk`), Matplotlib

## Project Structure

```text
numerical-analysis-workbench/
├── core/                  # Pure mathematical algorithms & AST parser
├── ui/                    # Tkinter GUI tabs & custom widgets
├── main.py                # Application entry point
└── requirements.txt       # Project dependencies