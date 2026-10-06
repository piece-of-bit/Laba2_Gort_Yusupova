# Лабораторная работа №2
# Прямые методы решения СЛАУ
# Вариант №5b и №17а
# Горт Александр и Юсупова Александра, НМТ-1,2

import math
from contextlib import redirect_stdout

# Глобальные параметры задачи

N = 4                 # размерность матрицы
EPS = 1e-12           # порог для сравнения с нулём (машинная точность)

OUTPUT_FILE = ".venv/lab2_output.txt"  # директория сохранения отчета


# Вывод матрицы
def matrix_print(A, name=""):
    if name:
        print(f"{name}:")
    for i in range(N):
        print(" ".join([f"{A[i][j]:.6f}" for j in range(N)]))
    print()


# Вывод вектора
def vector_print(v, name=""):
    if name:
        print(f"{name}: ", end="")
    print(" ".join([f"{v[i]:.6f}" for i in range(N)]))
    print()


# Копирование матрицы
def matrix_copy(src, dst):
    for i in range(N):
        for j in range(N):
            dst[i][j] = src[i][j]


# Создание единичной матрицы
def identity_matrix(I):
    for i in range(N):
        for j in range(N):
            I[i][j] = 1.0 if i == j else 0.0


# Умножение матриц
def matrix_multiply(A, B, C):
    for i in range(N):
        for j in range(N):
            C[i][j] = 0.0
            for k in range(N):
                C[i][j] += A[i][k] * B[k][j]

# Транспонирование
def transpose(A, AT):
    """Транспонирование: AT = A^T."""
    for i in range(N):
        for j in range(N):
            AT[j][i] = A[i][j]


# Кубическая норма (максимум из сумм модулей элементов по столбцам)
def norm1(A):
    mx = 0.0
    for j in range(N):
        s = 0.0
        for i in range(N):
            s += abs(A[i][j])
        if s > mx:
            mx = s
    return mx


# Октаэдрическая норма (максимум из сумм модулей элементов по строкам)
def norm_inf(A):
    mx = 0.0
    for i in range(N):
        s = 0.0
        for j in range(N):
            s += abs(A[i][j])
        if s > mx:
            mx = s
    return mx


# Метод вращений Якоби для поиска собственных значений (евклидова норма через A^T * A)
def jacobi_eigenvalues(S, lambda_max, lambda_min):
    max_iter = 1000
    # Копирование исходной матрицы
    A = [[S[i][j] for j in range(N)] for i in range(N)]

    for _ in range(max_iter):
        # Ищем максимальный по модулю внедиагональный элемент A[p][q]
        max_off_diag = 0.0
        p, q = 0, 1
        for i in range(N):
            for j in range(i + 1, N):
                if abs(A[i][j]) > max_off_diag:
                    max_off_diag = abs(A[i][j])
                    p, q = i, j

        # Если все внедиагональные малы — матрица почти диагональна, выходим
        if max_off_diag < EPS:
            break

        # Угол поворота, обнуляющий элемент A[p][q]
        app, aqq, apq = A[p][p], A[q][q], A[p][q]
        theta = 0.5 * math.atan2(2 * apq, aqq - app)
        c, s = math.cos(theta), math.sin(theta)

        # Обновление строк/столбцов, отличных от p и q
        for i in range(N):
            if i != p and i != q:
                aip, aiq = A[i][p], A[i][q]
                A[i][p] = A[p][i] = c * aip - s * aiq
                A[i][q] = A[q][i] = s * aip + c * aiq

        # Обновление диагональных элементов и обнуление A[p][q]
        A[p][p] = c * c * app - 2 * s * c * apq + s * s * aqq
        A[q][q] = s * s * app + 2 * s * c * apq + c * c * aqq
        A[p][q] = A[q][p] = 0.0

    # После итераций собственные значения стоят на диагонали
    lambda_max[0] = A[0][0]
    lambda_min[0] = A[0][0]
    for i in range(N):
        if A[i][i] > lambda_max[0]:
            lambda_max[0] = A[i][i]
        if A[i][i] < lambda_min[0]:
            lambda_min[0] = A[i][i]

    # Защита от деления на ноль при вычислении числа обусловленности
    if lambda_min[0] < EPS:
        lambda_min[0] = EPS


# Решение прямым ходом: решение L * y = b
def solve_Ly(L, b, y):
    for i in range(N):
        s = 0.0
        for j in range(i):
            s += L[i][j] * y[j]
        y[i] = (b[i] - s) / L[i][i]


# Решение обратным ходом: решение U * x = y
def solve_Ux(U, y, x):
    for i in range(N - 1, -1, -1):
        s = 0.0
        for j in range(i + 1, N):
            s += U[i][j] * x[j]
        x[i] = y[i] - s


# LU-разложение с выбором главного элемента по столбцу
def LU_decomposition(A_orig, L, U, P):
    # Копия исходной матрицы
    W = [[A_orig[i][j] for j in range(N)] for i in range(N)]

    # P — единичная матрица
    identity_matrix(P)

    # Обнуляем L и U; диагональ U сразу делаем единичной
    for i in range(N):
        for j in range(N):
            L[i][j] = 0.0
            U[i][j] = 0.0
        U[i][i] = 1.0

    # Cчетчик перестановок строк для знака определителя
    permutations = 0

    for k in range(N):
        # Выбор главного элемента по столбцу k в текущей матрице W
        pivot = k
        max_abs = abs(W[k][k])
        for i in range(k + 1, N):
            if abs(W[i][k]) > max_abs:
                max_abs = abs(W[i][k])
                pivot = i

        # Если весь столбец ниже диагонали — матрица вырождена
        if max_abs < EPS:
            return False, permutations

        # Перестановка строк (если нужна)
        if pivot != k:
            # Меняем строки в рабочей матрице W и в матрице перестановок P
            W[k], W[pivot] = W[pivot], W[k]
            P[k], P[pivot] = P[pivot], P[k]

            # Перестановка элементов L (в столбцах j < k)
            for j in range(k):
                L[k][j], L[pivot][j] = L[pivot][j], L[k][j]

            permutations += 1
            print(f"Шаг {k + 1}: перестановка строк {k} и {pivot}")

        # Формирование k-го столбца матрицы L
        L[k][k] = W[k][k]

        # Формирование k-й строки матрицы U (j > k)
        for j in range(k + 1, N):
            U[k][j] = W[k][j] / L[k][k]

        # Исключение нижележащих строк в столбце k
        for i in range(k + 1, N):
            L[i][k] = W[i][k]
            for j in range(k + 1, N):
                W[i][j] -= L[i][k] * U[k][j]
            W[i][k] = 0.0  # явно обнуляем

    # Проверка: диагональные элементы L не должны быть нулевыми
    for i in range(N):
        if abs(L[i][i]) < EPS:
            return False, permutations

    return True, permutations


# Основная обработка матрицы
def process_matrix(A_orig, name):
    print(f"\n{'=' * 50}")
    print(f"Обработка {name}")
    print('=' * 50)

    # Создание матриц
    L = [[0.0] * N for _ in range(N)]
    U = [[0.0] * N for _ in range(N)]
    P = [[0.0] * N for _ in range(N)]

    # LU-разложение с выбором главного элемента по столбцу
    success, permutations = LU_decomposition(A_orig, L, U, P)
    if not success:
        print("Матрица вырождена!")
        return

    # Ранг матрицы: количество ненулевых строк в U
    rank = 0
    for i in range(N):
        zero_row = True
        for j in range(N):
            if abs(U[i][j]) > EPS:
                zero_row = False
                break
        if not zero_row:
            rank += 1

    # Определитель (s — число перестановок строк)
    sign = 1.0 if permutations % 2 == 0 else -1.0
    det = sign
    for i in range(N):
        det *= L[i][i]

    print(f"\nОпределитель: {det:.6f}")
    print(f"Ранг: {rank}")

    # Печать исходных и промежуточных матриц
    matrix_print(A_orig, "Исходная матрица A")
    matrix_print(P,      "Матрица перестановок P")
    matrix_print(L,      "Матрица L")
    matrix_print(U,      "Матрица U")

    # Обратная матрица через n решений СЛАУ с единичными векторами
    A_inv = [[0.0] * N for _ in range(N)]
    I = [[0.0] * N for _ in range(N)]
    identity_matrix(I)

    for col in range(N):
        # Единичный вектор
        e = [I[i][col] for i in range(N)]

        # Учитываем перестановку
        Pb = [0.0] * N
        for i in range(N):
            for k in range(N):
                Pb[i] += P[i][k] * e[k]

        # Решаем L*y = Pb, затем U*x = y
        y = [0.0] * N
        x_col = [0.0] * N
        solve_Ly(L, Pb, y)
        solve_Ux(U, y, x_col)

        # Записываем результат в col-й столбец обратной матрицы
        for i in range(N):
            A_inv[i][col] = x_col[i]

    matrix_print(A_inv, "Обратная матрица A^-1")

    # Числа обусловленности в трёх нормах
    cond1   = norm1(A_orig)   * norm1(A_inv)     # кубическая
    cond_inf = norm_inf(A_orig) * norm_inf(A_inv) # октаэдрическая

    # Евклидова
    AT = [[0.0] * N for _ in range(N)]
    ATA = [[0.0] * N for _ in range(N)]
    transpose(A_orig, AT)
    matrix_multiply(AT, A_orig, ATA)

    lambda_max = [0.0]
    lambda_min = [0.0]
    jacobi_eigenvalues(ATA, lambda_max, lambda_min)

    cond2 = math.sqrt(lambda_max[0] / lambda_min[0])

    print("Числа обусловленности:")
    print(f"Кубическая норма (L1): {cond1:.6f}")
    print(f"Октаэдрическая норма: {cond_inf:.6f}")
    print(f"Евклидова норма (L2): {cond2:.6f}")

    # Формируем вектор b
    x_true = [1.0, 2.0, 3.0, 4.0]
    b = [0.0] * N
    for i in range(N):
        for j in range(N):
            b[i] += A_orig[i][j] * x_true[j]

    vector_print(b, "Вектор b (A*x)")

    # Решаем систему A*x = b
    Pb = [0.0] * N
    for i in range(N):
        for k in range(N):
            Pb[i] += P[i][k] * b[k]

    y = [0.0] * N
    x_sol = [0.0] * N
    solve_Ly(L, Pb, y)
    solve_Ux(U, y, x_sol)

    vector_print(x_sol, "Решение x из LU")

    # Проверка невязки r
    res = [0.0] * N
    for i in range(N):
        s = 0.0
        for j in range(N):
            s += A_orig[i][j] * x_sol[j]
        res[i] = s - b[i]

    res_norm = max(abs(r) for r in res)
    print(f"Невязка Ax - b: {res_norm:.6e}")

    # Проверка разложения
    PA = [[0.0] * N for _ in range(N)]
    LU = [[0.0] * N for _ in range(N)]
    matrix_multiply(P, A_orig, PA)
    matrix_multiply(L, U, LU)

    diff1 = 0.0
    for i in range(N):
        for j in range(N):
            diff1 = max(diff1, abs(PA[i][j] - LU[i][j]))
    print(f"Проверка PA - LU: {diff1:.6e}")

    # Проверка корректности обращения матрицы
    AA_inv = [[0.0] * N for _ in range(N)]
    I_prov = [[0.0] * N for _ in range(N)]
    matrix_multiply(A_orig, A_inv, AA_inv)
    identity_matrix(I_prov)

    diff2 = 0.0
    for i in range(N):
        for j in range(N):
            diff2 = max(diff2, abs(AA_inv[i][j] - I_prov[i][j]))
    print(f"Проверка A*A^-1 - I: {diff2:.6e}")
    print()


def main():
    # Исходная матрица A варианта №5b
    A1 = [
        [8.4,  6.1, -1.9,  0.4],
        [0.1, -0.3, -2.5, -8.7],
        [4.9,  9.5, -7.9,  1.8],
        [0.2,  1.9, -9.9, -8.0]
    ]
    # Исходная матрица A варианта №17a
    A2 = [
        [-7.7,  -2.8, 0.4,  0.1],
        [4.6,   3.1,  7.9,  4.4],
        [-5.0,  1.1, -9.6,  6.7],
        [4.4,   7.4, -0.7, -3.7]
    ]

    # Вывод программы в файл отчёта
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        with redirect_stdout(f):
            process_matrix(A1, "Матрица A1")
            process_matrix(A2, "Матрица A2")

    print(f"Результаты сохранены в файл: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()