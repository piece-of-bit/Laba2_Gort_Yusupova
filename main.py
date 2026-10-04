import math
from contextlib import redirect_stdout

N = 4
EPS = 1e-12

OUTPUT_FILE = ".venv/lab2_output.txt"


def matrix_print(A, name=""):
    if name:
        print(f"{name}:")
    for i in range(N):
        print(" ".join([f"{A[i][j]:.6f}" for j in range(N)]))
    print()


def vector_print(v, name=""):
    if name:
        print(f"{name}: ", end="")
    print(" ".join([f"{v[i]:.6f}" for i in range(N)]))
    print()


def matrix_copy(src, dst):
    for i in range(N):
        for j in range(N):
            dst[i][j] = src[i][j]


def identity_matrix(I):
    for i in range(N):
        for j in range(N):
            I[i][j] = 1.0 if i == j else 0.0


def matrix_multiply(A, B, C):
    for i in range(N):
        for j in range(N):
            C[i][j] = 0.0
            for k in range(N):
                C[i][j] += A[i][k] * B[k][j]


def transpose(A, AT):
    for i in range(N):
        for j in range(N):
            AT[j][i] = A[i][j]


def norm1(A):
    mx = 0.0
    for j in range(N):
        s = 0
        for i in range(N):
            s += abs(A[i][j])
        if s > mx:
            mx = s
    return mx


def norm_inf(A):
    mx = 0.0
    for i in range(N):
        s = 0
        for j in range(N):
            s += abs(A[i][j])
        if s > mx:
            mx = s
    return mx


def jacobi_eigenvalues(S, lambda_max, lambda_min):
    max_iter = 1000
    A = [[S[i][j] for j in range(N)] for i in range(N)]

    for _ in range(max_iter):
        max_off_diag = 0.0
        p, q = 0, 1
        for i in range(N):
            for j in range(i + 1, N):
                if abs(A[i][j]) > max_off_diag:
                    max_off_diag = abs(A[i][j])
                    p, q = i, j

        if max_off_diag < EPS:
            break

        app, aqq, apq = A[p][p], A[q][q], A[p][q]
        theta = 0.5 * math.atan2(2 * apq, aqq - app)
        c, s = math.cos(theta), math.sin(theta)

        for i in range(N):
            if i != p and i != q:
                aip, aiq = A[i][p], A[i][q]
                A[i][p] = A[p][i] = c * aip - s * aiq
                A[i][q] = A[q][i] = s * aip + c * aiq

        A[p][p] = c * c * app - 2 * s * c * apq + s * s * aqq
        A[q][q] = s * s * app + 2 * s * c * apq + c * c * aqq
        A[p][q] = A[q][p] = 0.0

    lambda_max[0] = A[0][0]
    lambda_min[0] = A[0][0]
    for i in range(N):
        if A[i][i] > lambda_max[0]:
            lambda_max[0] = A[i][i]
        if A[i][i] < lambda_min[0]:
            lambda_min[0] = A[i][i]

    if lambda_min[0] < EPS:
        lambda_min[0] = EPS


def solve_Ly(L, b, y):
    for i in range(N):
        s = 0.0
        for j in range(i):
            s += L[i][j] * y[j]
        y[i] = (b[i] - s) / L[i][i]


def solve_Ux(U, y, x):
    for i in range(N - 1, -1, -1):
        s = 0.0
        for j in range(i + 1, N):
            s += U[i][j] * x[j]
        x[i] = y[i] - s


def LU_decomposition(A_orig, L, U, P):
    A = [[A_orig[i][j] for j in range(N)] for i in range(N)]

    identity_matrix(P)
    for i in range(N):
        for j in range(N):
            L[i][j] = 0.0
            U[i][j] = 0.0
        U[i][i] = 1.0

    permutations = 0

    for k in range(N):
        pivot = k
        for i in range(k + 1, N):
            if abs(A[i][k]) > abs(A[pivot][k]):
                pivot = i

        if pivot != k:
            A[k], A[pivot] = A[pivot], A[k]
            P[k], P[pivot] = P[pivot], P[k]
            if k > 0:
                for j in range(k):
                    L[k][j], L[pivot][j] = L[pivot][j], L[k][j]
            permutations += 1
            print(f"Шаг {k + 1}: перестановка строк {k} и {pivot}")

        for i in range(k, N):
            s = 0.0
            for s_idx in range(k):
                s += L[i][s_idx] * U[s_idx][k]
            L[i][k] = A[i][k] - s

        for j in range(k + 1, N):
            s = 0.0
            for s_idx in range(k):
                s += L[k][s_idx] * U[s_idx][j]
            U[k][j] = (A[k][j] - s) / L[k][k]

    for i in range(N):
        if abs(L[i][i]) < EPS:
            return False, permutations

    return True, permutations


def process_matrix(A_orig, name):
    print(f"\n{'=' * 50}")
    print(f"Обработка {name}")
    print('=' * 50)

    L = [[0.0] * N for _ in range(N)]
    U = [[0.0] * N for _ in range(N)]
    P = [[0.0] * N for _ in range(N)]

    success, permutations = LU_decomposition(A_orig, L, U, P)

    if not success:
        print("Матрица вырождена!")
        return

    rank = 0
    for i in range(N):
        zero_row = True
        for j in range(N):
            if abs(U[i][j]) > EPS:
                zero_row = False
                break
        if not zero_row:
            rank += 1

    sign = 1.0 if permutations % 2 == 0 else -1.0
    det = sign
    for i in range(N):
        det *= L[i][i]

    print(f"\nОпределитель: {det:.6f}")
    print(f"Ранг: {rank}")

    matrix_print(A_orig, "Исходная матрица A")
    matrix_print(P, "Матрица перестановок P")
    matrix_print(L, "Матрица L")
    matrix_print(U, "Матрица U")

    A_inv = [[0.0] * N for _ in range(N)]
    I = [[0.0] * N for _ in range(N)]
    identity_matrix(I)

    for col in range(N):
        e = [I[i][col] for i in range(N)]

        Pb = [0.0] * N
        for i in range(N):
            for k in range(N):
                Pb[i] += P[i][k] * e[k]

        y = [0.0] * N
        x_col = [0.0] * N
        solve_Ly(L, Pb, y)
        solve_Ux(U, y, x_col)

        for i in range(N):
            A_inv[i][col] = x_col[i]

    matrix_print(A_inv, "Обратная матрица A^-1")

    cond1 = norm1(A_orig) * norm1(A_inv)
    cond_inf = norm_inf(A_orig) * norm_inf(A_inv)

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

    x_true = [1.0, 2.0, 3.0, 4.0]
    b = [0.0] * N
    for i in range(N):
        for j in range(N):
            b[i] += A_orig[i][j] * x_true[j]

    vector_print(b, "Вектор b (A*x)")

    Pb = [0.0] * N
    for i in range(N):
        for k in range(N):
            Pb[i] += P[i][k] * b[k]

    y = [0.0] * N
    x_sol = [0.0] * N
    solve_Ly(L, Pb, y)
    solve_Ux(U, y, x_sol)

    vector_print(x_sol, "Решение x из LU")

    res = [0.0] * N
    for i in range(N):
        s = 0.0
        for j in range(N):
            s += A_orig[i][j] * x_sol[j]
        res[i] = s - b[i]

    res_norm = max(abs(r) for r in res)
    print(f"Невязка Ax - b: {res_norm:.6e}")

    PA = [[0.0] * N for _ in range(N)]
    LU = [[0.0] * N for _ in range(N)]
    matrix_multiply(P, A_orig, PA)
    matrix_multiply(L, U, LU)

    diff1 = 0.0
    for i in range(N):
        for j in range(N):
            diff1 = max(diff1, abs(PA[i][j] - LU[i][j]))
    print(f"Проверка PA - LU: {diff1:.6e}")

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
    A1 = [
        [8.4, 6.1, -1.9, 0.4],
        [0.1, -0.3, -2.5, -8.7],
        [4.9, 9.5, -7.9, 1.8],
        [0.2, 1.9, -9.9, -8.0]
    ]

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        with redirect_stdout(f):
            process_matrix(A1, "Матрица A1")

    print(f"Результаты сохранены в файл: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()