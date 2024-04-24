import numpy as np

eps = pow(10, -10)


def memorareEco(data):
    size = data[0][0]
    matrix = [[] for _ in range(size)]
    
    for value, row, col in data[1:]:
        matrix[row].append((value, col))
        
    return matrix


def memorareCSR(data):
    values = []
    ind_col = []
    inceput_linii = []
    for value, row, col in data[1:]:
        if len(inceput_linii) <= row:
            inceput_linii.append(len(values))
        values.append(value)
        ind_col.append(col)
    inceput_linii.append(len(values))
    return values, ind_col, inceput_linii


def cititFisier(nume_fisier):
    data = []
    with open(nume_fisier, "r") as f:
        lines = f.readlines()
        for line in lines:
            elements = line.split(",")
            elements = [elem.strip() for elem in elements]
            if len(elements) == 1 and elements[0]: 
                data.append((int(elements[0]), None, None))
            elif len(elements) == 3:
                value = float(elements[0])
                row = int(elements[1])
                col = int(elements[2])
                data.append((value, row, col))
    return data


def cititVector(fisier):
    vector = []
    with open(fisier, "r") as file:
        n = int(file.readline().strip())
        for _ in range(n):
            element = float(file.readline().strip())
            vector.append(element)
    return vector


def diagVerif(matrix, n):
    check = True
    for i in range(n):
        if not any(col == i for _, col in matrix[i]):
            print(f"Elementul de pe diagonala la indexul {i} este zero sau lipsă")
            check = False
    if not check:
        print("MATRICEA NU ARE ELEMENTE EGALE CU 0 SAU LIPSA PE DIAGONALA PRINCIPALA")


def gaussSeidel(A, b, max_iterations=1000):
    n = len(b)
    x = [0.0] * n
    for it_count in range(max_iterations):
        convergence = True
        for i in range(n):
            old_xi = x[i]
            s1 = 0.0
            s2 = 0.0
            for val, j in A[i]:
                if j < i and abs(val) > eps:
                    s1 += val * x[j]
                elif j > i and abs(val) > eps:
                    s2 += val * x[j]
            x[i] = (b[i] - s1 - s2) / next(val for val, j in A[i] if j == i)
            if abs(x[i] - old_xi) >= eps:
                convergence = False
        if convergence:
            break
        if any(x[i] is None or abs(x[i]) == float('inf') for i in range(n)):
            return None, it_count + 1

    return x, it_count + 1


def gaussSeidel2(values, ind_col, inceput_linii, b, max_iter=1000):
    n = len(b)
    x = [0] * n
    for it_count in range(max_iter):
        converged = True

        for i in range(n):
            sum1 = 0.0 
            sum2 = 0.0
            diag_value = 0.0

            start = inceput_linii[i]
            end = inceput_linii[i + 1]
            for j in range(start, end):
                if ind_col[j] < i:
                    sum1 += values[j] * x[ind_col[j]]
                elif ind_col[j] > i:
                    sum2 += values[j] * x[ind_col[j]]
                else:
                    diag_value = values[j]

            new_xi = (b[i] - sum1 - sum2) / diag_value
            if abs(new_xi - x[i]) >= eps:
                converged = False
            x[i] = new_xi  

        if converged:
            break
        if any(x[i] is None or abs(x[i]) == float('inf') for i in range(n)):
            return None, it_count + 1
    return x, it_count + 1

def calculareNorma(A, x_gs, b):
    n = len(b)
    Ax_gs = np.zeros(n)

    for i in range(n):
        Ax_gs[i] = sum(val * x_gs[j] for val, j in A[i])

    norm = np.linalg.norm(Ax_gs - np.array(b), ord=np.inf)
    return norm


def citire():
    fisier_corect = False
    while not fisier_corect:
        numar_fisier = int(input("Introduceti un numar pentru fisier (De la 1 pana la 5): "))
        nume_fisiere = {
            1: ("a_1.txt", "b_1.txt"),
            2: ("a_2.txt", "b_2.txt"),
            3: ("a_3.txt", "b_3.txt"),
            4: ("a_4.txt", "b_4.txt"),
            5: ("a_5.txt", "b_5.txt")
        }
        if numar_fisier in nume_fisiere:
            nume_fisier1, nume_fisier2 = nume_fisiere[numar_fisier]
            fisier_corect = True
        else:
            print("Numarul introdus nu este valid. Trebuie să fie între 1 și 5.")

    return nume_fisier1, nume_fisier2


def main():
    nume_fisier1, nume_fisier2 = citire()
    data = cititFisier(nume_fisier1)
    b = cititVector(nume_fisier2)
    A = memorareEco(data)
    values, ind_col, inceput_linii = memorareCSR(data)
    diagVerif(A, data[0][0])
    solution, iterations = gaussSeidel(A, b)
    solution2, iterations2 = gaussSeidel2(values, ind_col, inceput_linii, b)
    print("A(Memorare 1):")
    for each in range(0, 5):
        print(A[each])
    print("\n")
    print("A(Memorare 2):")
    print(f"Valori", values[:5])
    print(f"Indici coloane", ind_col[:5])
    print(f"Inceput linii", inceput_linii[:5])
    print("\n")
    print("Memorare 1 Gauss Seidel:")
    if solution is not None:
        print("Solutie:", solution[-10:-1])
        norma = calculareNorma(A, solution, b)
        print("Norma:", norma,";","Numar iteratii:", iterations)
    else:
        print("Divergenta")
    print("\n")
    print("Memorare 2 Gauss Seidel:")
    if solution2 is not None:
        print("Solutie:", solution2[-10:-1])
        norma = calculareNorma(A, solution2, b)
        print("Norma:", norma,";","Numar de iteratii", iterations2)
    else:
        print("Divergenta")

if __name__ == "__main__":
    main()