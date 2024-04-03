import math
import copy
import numpy as np

def productMatrix(A, x):
    rez = [0 for i in range(0, len(A))]
    for i in range(0, len(A)):
        for j in range(0, len(A[0])):
            rez[i] += A[i][j] * x[j]
    return rez

def Euclid(x):
    return math.sqrt(sum(x[i]**2 for i in range(0, len(x))))

def decr_vect(a, b):
    x = [a[i]-b[i] for i in range(0, len(a))]
    return x

M = [[0, 1, 0],
     [1, 0, 0],
     [0, 0, 1]]

s = [1, 2, 3]

'''
A = [[0, 0, 4, 1],
     [1, 2, 3, 2],
     [0, 1, 2, 3],
     [1, 1, 1, 1]]

s = [3, 2, 1, 4]
'''

def displayMatrix(A):
  # display the matrix formatted so that each row is on a separate line, and it is easy to see the elements
  for i in range(len(A)):
    print(A[i])
   

epsilon = 0.000000000000001

def getB(a, s):
    n = len(a)
    b = [0 for i in range(0, n)]

    for i in range(0, n):
        b[i] = sum([s[j] * a[i][j] for j in range(0, n)])

    return b

b = getB(M, s)
print("Exercitiul I")
print("Vectorul b:")
print(b)
print()

def getQR(a, b):
    n = len(a)
    QTemp = []
    for i in range(n):
        row = []
        for j in range(n):
            row.append(0)
        QTemp.append(row)
        
    for i in range(0, n):
        QTemp[i][i] = 1    
        
    for r in range(0, n-1):

        sigma = sum([a[i][r]**2 for i in range(r, n)])

        if (sigma <= epsilon):
            return None
        
        k = math.sqrt(sigma)

        if (a[r][r] > 0):
            k = -k
        
        beta = sigma - k * a[r][r]

        u = [0 for i in range(0, n)]
        u[r] = a[r][r] - k
        for i in range(r+1, n):
            u[i] = a[i][r]

        # transformare Householder
        for j in range(r+1, n):
            gamma = sum([u[i]*a[i][j] for i in range(r, n)]) / beta

            for i in range(r, n):
                a[i][j] = a[i][j] - gamma * u[i]
        
        a[r][r] = k
        for i in range(r+1, n):
            a[i][r] = 0

        gamma = sum([u[i]*b[i] for i in range(r, n)]) / beta

        for i in range(r, n):
            b[i] = b[i] - gamma * u[i]
        
        for j in range(n):
            gamma = sum([u[i]*QTemp[i][j] for i in range(r, n)]) / beta

            for i in range(r, n):
                QTemp[i][j] = QTemp[i][j] - gamma * u[i]  

    R = a
    QT = QTemp
    QTb = b
    return R, QT, QTb

R, QT, QTb = getQR(M, b)
print("Exercitiul II")
print("Matricea R:")
displayMatrix(R)
print()

print("Matricea QTransp:")
displayMatrix(QT)
print()

print("QTransp * b:")
displayMatrix(QTb)
print()

def solveEQ(a, b):
    n = len(a)
    x = [0 for i in range(0, n)]
    for i in range(n-1, -1, -1):
        sum = 0
        for j in range(i, n):
            sum += a[i][j]*x[j]
        x[i] = (b[i] - sum) / a[i][i]
    return x

xHH = solveEQ(R, QTb)
print("Exercitiul III")
print("x:")
print(xHH)

AQR = np.array(M)
bQR = np.array(b)
QQR, RQR = np.linalg.qr(AQR)
pQR = np.dot(QQR.T, bQR)
xQR = np.dot(np.linalg.inv(RQR), pQR)
xQR = np.ndarray.tolist(xQR)

print("xQR:")
print(xQR)
print("||xQR - XHH||")
print(Euclid(decr_vect(xQR, xHH)), '\n')

print("Exercitiul IV")
print("||Ainit * xHH - binit||")
print(Euclid(decr_vect(productMatrix(M, xHH),b)), "\n")
print("||Ainit * xQR - binit||")
print(Euclid(decr_vect(productMatrix(M, xQR), b)),  "\n")
print("||xHH - s|| / ||s||")
print(Euclid(decr_vect(xHH, s)) / Euclid(s), "\n")
print("||xQR - s|| / ||s||")
print(Euclid(decr_vect(xQR, s)) / Euclid(s) ,"\n")

def invMatrix(a):
    n = len(a)
    Ainv = [[0 for j in range(0, n)] for i in range(0, n)]
    for j in range(0, n):
        b = [ 1 if i == n - j - 1 else 0 for i in range(0, n)]
        R, QT, QTb = getQR(a, b)
        x = solveEQ(R, QTb)
        for i in range(0, n):
            Ainv[i][j] = x[i]
    
    return Ainv

Ainv = invMatrix(M)
print("Exercitiul V")
print("Inversa matricii A:")
displayMatrix(Ainv)