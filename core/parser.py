import ast
import operator
import numpy as np
from scipy.special import erf

def laplace_probability_integral(x):
    """Phi(x) = 1/sqrt(2*pi) * integral from 0 to x exp(-t^2/2) dt."""
    return 0.5 * erf(np.asarray(x, dtype=float) / np.sqrt(2.0))

_ALLOWED_FUNCTIONS = {
    "sin": np.sin, "cos": np.cos, "tan": np.tan, "exp": np.exp,
    "ln": np.log, "log": np.log, "lg": np.log10, "log10": np.log10,
    "sqrt": np.sqrt, "abs": np.abs, "erf": erf, "Phi": laplace_probability_integral,
}
_ALLOWED_CONSTANTS = {"pi": np.pi, "e": np.e}
_ALLOWED_BINARY = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
                   ast.Div: operator.truediv, ast.Pow: operator.pow}
_ALLOWED_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}

def evaluate_expression(expression, x):
    """Безпечно обчислює прості математичні вирази, не виконуючи довільний Python-код."""
    tree = ast.parse(expression.replace("^", "**"), mode="eval")

    def evaluate(node):
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if isinstance(node, ast.Name):
            if node.id == "x":
                return x
            if node.id in _ALLOWED_CONSTANTS:
                return _ALLOWED_CONSTANTS[node.id]
            raise ValueError(f"Невідоме ім'я «{node.id}». Використовуй x, pi, e або дозволені функції.")
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINARY:
            return _ALLOWED_BINARY[type(node.op)](evaluate(node.left), evaluate(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARY:
            return _ALLOWED_UNARY[type(node.op)](evaluate(node.operand))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _ALLOWED_FUNCTIONS:
            if len(node.args) != 1 or node.keywords:
                raise ValueError(f"Функція {node.func.id} має отримувати один аргумент, наприклад {node.func.id}(x).")
            return _ALLOWED_FUNCTIONS[node.func.id](evaluate(node.args[0]))
        raise ValueError(
            "Не вдалося розібрати функцію. Дозволені операції +, -, *, /, ** та функції "
            "sin, cos, tan, exp, ln, log, lg, log10, sqrt, abs, erf, Phi."
        )

    return evaluate(tree)