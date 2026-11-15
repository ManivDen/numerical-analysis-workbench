def print_matrix(matrix, label):
    print(f"\n{label}:")
    for row in matrix:
        formatted = [f"{val:8.3f}" for val in row[:-1]]
        print(f"[ {' '.join(formatted)} | {row[-1]:8.3f} ]")
    print()

def solve_gaussian(a_matrix, b_vector):
    n = len(b_vector)
    aug = [a_matrix[i][:] + [b_vector[i]] for i in range(n)]
    print_matrix(aug, "початкова розширена матриця")

    for i in range(n):
        print(f"\n--- етап {i + 1} ---")
        if abs(aug[i][i]) < 1e-12:
            raise ValueError(f"на діагоналі в рядку {i + 1} нуль, метод без перестановки не спрацює.")

        pivot_val = aug[i][i]
        if abs(pivot_val - 1.0) > 1e-12:
            print(f"крок: ділимо рядок {i + 1} на {pivot_val:.3f}, щоб отримати 1 на діагоналі")
            for col in range(i, n + 1):
                aug[i][col] /= pivot_val

        for j in range(i + 1, n):
            factor = aug[j][i]
            if abs(factor) > 1e-12:
                print(f"крок: віднімаємо від рядка {j + 1} рядок {i + 1}, помножений на {factor:.3f}")
                for col in range(i, n + 1):
                    aug[j][col] -= factor * aug[i][col]
        print_matrix(aug, f"результат після етапу {i + 1}")

    print("\n--- зворотний хід ---")
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        rhs = aug[i][-1]
        terms = []
        sub_sum = 0.0
        for j in range(i + 1, n):
            val = x[j]
            sub_sum += aug[i][j] * val
            terms.append(f"({aug[i][j]:.3f} * {val:.3f})")

        if terms:
            terms_str = " - ".join(terms)
            print(f"\nкрок для x{i + 1}:")
            print(f"  формула: x{i + 1} = {rhs:.3f} - [{terms_str}]")
            print(f"  обчислення: x{i + 1} = {rhs:.3f} - {sub_sum:.3f}")
        else:
            print(f"\nкрок для x{i + 1}:")
            print(f"  формула: x{i + 1} = {rhs:.3f}")

        x[i] = rhs - sub_sum
        print(f"  результат: x{i + 1} = {x[i]:.4f}")
    return x

def check_diagonal_dominance(a_matrix):
    n = len(a_matrix)
    is_dominant = True
    print("\nкрок 1. перевірка на діагональне переважання")
    for i in range(n):
        row_sum = sum(abs(a_matrix[i][j]) for j in range(n) if i != j)
        diag = abs(a_matrix[i][i])
        if diag > row_sum:
            print(f"рядок {i+1}: |{a_matrix[i][i]}| > {row_sum:.4f} – умова виконується")
        else:
            print(f"рядок {i+1}: |{a_matrix[i][i]}| <= {row_sum:.4f} – умова не виконується")
            is_dominant = False
    if is_dominant:
        print("діагональне переважання строге, система гарантовано збігається.")
    else:
        print("увага: переважання немає, метод може розбігтися.")
    return is_dominant

def solve_jacobi(a_matrix, b_vector, epsilon, max_iter=100):
    n = len(a_matrix)
    check_diagonal_dominance(a_matrix)

    print("\nкрок 2. ізоляція невідомих")
    c_matrix = [[0.0] * n for _ in range(n)]
    d_vector = [0.0] * n

    for i in range(n):
        d_vector[i] = b_vector[i] / a_matrix[i][i]
        for j in range(n):
            if i != j:
                c_matrix[i][j] = -a_matrix[i][j] / a_matrix[i][i]

    for i in range(n):
        equation = f"x{i+1} = {d_vector[i]:.4f}"
        for j in range(n):
            if i != j:
                sign = "+" if c_matrix[i][j] >= 0 else "-"
                equation += f" {sign} {abs(c_matrix[i][j]):.4f}*x{j+1}"
        print(equation)

    print("\nкрок 3. нульова ітерація")
    x = d_vector[:]
    for i in range(n):
        print(f"x{i+1}^(0) = {x[i]:.4f}")

    print("\nкрок 4 та 5. ітерації та контроль похибки")
    for k in range(max_iter):
        print(f"\n--- ітерація {k+1} ---")
        x_new = [0.0] * n
        errors = [0.0] * n
        for i in range(n):
            calc_str = f"x{i+1}^({k+1}) = {d_vector[i]:.4f}"
            s = 0.0
            for j in range(n):
                if i != j:
                    sign = "+" if c_matrix[i][j] >= 0 else "-"
                    calc_str += f" {sign} {abs(c_matrix[i][j]):.4f}*{x[j]:.4f}"
                    s += c_matrix[i][j] * x[j]
            x_new[i] = d_vector[i] + s
            errors[i] = abs(x_new[i] - x[i])
            print(f"{calc_str} = {x_new[i]:.4f}")

        max_err = max(errors)
        print("похибки на цьому кроці:")
        for i in range(n):
            print(f"|x{i+1}^({k+1}) - x{i+1}^({k})| = {errors[i]:.6f}")
        print(f"максимальна похибка: {max_err:.6f}")

        if max_err <= epsilon:
            print(f"\nпохибка {max_err:.6f} <= {epsilon}, результат {k+1} ітерації.")
            return x_new
        else:
            print(f"похибка {max_err:.6f} > {epsilon}, рухаємося далі.")
        x = x_new[:]
    print("\nліміт ітерацій вичерпано.")
    return x