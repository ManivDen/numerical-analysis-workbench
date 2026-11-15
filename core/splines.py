def make_grid(x0, h, n):
    """Рівновіддалені вузли x_i = x_0 + i*h, i = 0..n."""
    return x0 + h * np.arange(n + 1, dtype=float)


def linear_bspline(x, k, x0, h):
    """B_{1,k}: m=2, d=1. Носій [x_k, x_{k+2}], B(x_{k+1}) = 1."""
    t = (np.asarray(x, dtype=float) - (x0 + k * h)) / h
    y = np.zeros_like(t)
    m1 = (t >= 0) & (t < 1)
    m2 = (t >= 1) & (t <= 2)
    y[m1] = t[m1]
    y[m2] = 2.0 - t[m2]
    return y


def cubic_bspline(x, k, x0, h):
    """B_{3,k}: m=4, d=1. Носій [x_k, x_{k+4}], чотири кубічні многочлени.

    t = (x - x_k)/h:
      [0,1): t^3/6
      [1,2): (-3t^3 + 12t^2 - 12t + 4)/6
      [2,3): ( 3t^3 - 24t^2 + 60t - 44)/6
      [3,4]: (4 - t)^3/6
    Значення у вузлах: 0, 1/6, 2/3, 1/6, 0. Клас гладкості C^2.
    """
    t = (np.asarray(x, dtype=float) - (x0 + k * h)) / h
    y = np.zeros_like(t)
    m1 = (t >= 0) & (t < 1)
    m2 = (t >= 1) & (t < 2)
    m3 = (t >= 2) & (t < 3)
    m4 = (t >= 3) & (t <= 4)
    y[m1] = t[m1] ** 3 / 6.0
    y[m2] = (-3 * t[m2] ** 3 + 12 * t[m2] ** 2 - 12 * t[m2] + 4) / 6.0
    y[m3] = (3 * t[m3] ** 3 - 24 * t[m3] ** 2 + 60 * t[m3] - 44) / 6.0
    y[m4] = (4.0 - t[m4]) ** 3 / 6.0
    return y

METHODS = {
    "linear": dict(func=linear_bspline, width=2, tag=1, name="Лінійний B-сплайн (m=2)",
                   pieces=["t", "2 - t"]),
    "cubic": dict(func=cubic_bspline, width=4, tag=3, name="Кубічний B-сплайн (m=4)",
                  pieces=["t^3 / 6",
                          "(-3t^3 + 12t^2 - 12t + 4) / 6",
                          "(3t^3 - 24t^2 + 60t - 44) / 6",
                          "(4 - t)^3 / 6"]),
}

SPLINE_COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple",
                 "tab:brown", "tab:pink", "tab:olive", "tab:cyan", "tab:gray"]