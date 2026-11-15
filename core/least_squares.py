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