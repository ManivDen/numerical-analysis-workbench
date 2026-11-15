import numpy as np

def lagrange_interpolation(x_nodes, y_nodes, x_values):
    x_nodes = np.asarray(x_nodes, dtype=float)
    y_nodes = np.asarray(y_nodes, dtype=float)
    x_values = np.asarray(x_values, dtype=float)
    result = np.zeros_like(x_values, dtype=float)
    for i in range(len(x_nodes)):
        basis = np.ones_like(x_values, dtype=float)
        for j in range(len(x_nodes)):
            if i != j:
                basis *= (x_values - x_nodes[j]) / (x_nodes[i] - x_nodes[j])
        result += y_nodes[i] * basis
    return result


def newton_forward_interpolation(x_nodes, y_nodes, h, x_values):
    x_nodes = np.asarray(x_nodes, dtype=float)
    y_nodes = np.asarray(y_nodes, dtype=float)
    x_values = np.asarray(x_values, dtype=float)

    first_differences = [y_nodes[0]]
    row = y_nodes.copy()
    while len(row) > 1:
        row = np.diff(row)
        first_differences.append(row[0])

    t = (x_values - x_nodes[0]) / h
    result = np.zeros_like(x_values, dtype=float) + first_differences[0]
    product = np.ones_like(x_values, dtype=float)
    for k in range(1, len(first_differences)):
        product *= (t - (k - 1)) / k
        result += first_differences[k] * product
    return result


def newton_divided_differences(x_nodes, y_nodes):
    x_nodes = np.asarray(x_nodes, dtype=float)
    coefficients = np.asarray(y_nodes, dtype=float).copy()
    count = len(x_nodes)
    for order in range(1, count):
        coefficients[order:] = (
            coefficients[order:] - coefficients[order - 1:-1]
        ) / (x_nodes[order:] - x_nodes[:-order])
    return coefficients


def newton_divided_interpolation(x_nodes, coefficients, x_values):
    x_nodes = np.asarray(x_nodes, dtype=float)
    x_values = np.asarray(x_values, dtype=float)
    coefficients = np.asarray(coefficients, dtype=float)
    result = np.zeros_like(x_values, dtype=float) + coefficients[-1]
    for k in range(len(coefficients) - 2, -1, -1):
        result = coefficients[k] + (x_values - x_nodes[k]) * result
    return result

def chebyshev_roots(a, b, node_count):
    k = np.arange(1, node_count + 1, dtype=float)
    roots = np.cos((2 * k - 1) * np.pi / (2 * node_count))
    mapped = (a + b) / 2 + (b - a) / 2 * roots
    return np.sort(mapped)