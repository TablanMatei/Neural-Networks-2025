import pathlib
import math
import copy
#TASK 1: Parsing the System of Equations
'''
Citit linie cu linie si dupa parsat
strip ca sa scoatem backslash n
split la = ca sa obtinem matricea B
Parcurs caracter cu caracter si construit coeficient
pentru fiecare x,y,z
Tratat caz cand coeficnet=1 sau coeficient=0
'''

def load_system(path: pathlib.Path) -> tuple[list[list[float]], list[float]]:
    with path.open('r') as f:
        lines = f.readlines()

    index=0
    a=[[],[],[]]
    b=[]

    for line in lines:
        line = line.strip()
        line = line.split("=")
        b.append(int(line[1]))

        pozitiv = True
        numar = 0
        exist_x = False
        exist_y = False
        exist_z = False
        #print(line[0])
        for ch in line[0]:
            if ch=='-': pozitiv=False
            if ch.isdigit(): numar=numar*10 + int(ch)
            if ch=='x':
                exist_x=True
                if numar == 0: numar = 1
                if not pozitiv: numar *=- 1
                a[index].append(numar)
                numar = 0
                pozitiv = True
            if ch == 'y':
                exist_y = True
                if numar==0: numar=1
                if not pozitiv: numar *= - 1
                if not exist_x:
                    a[index].append(0)
                    exist_x = True
                a[index].append(numar)
                numar = 0
                pozitiv = True
            if ch=='z':
                exist_z=True
                if numar == 0: numar = 1
                if not pozitiv: numar *= - 1
                if not exist_x:
                    a[index].append(0)
                    exist_x = True
                if not exist_y:
                    a[index].append(0)
                    exist_y = True
                a[index].append(numar)
                numar = 0
                pozitiv = True
        if not exist_y: a[index].append(0)
        if not exist_z: a[index].append(0)
        index += 1

    return a, b

# TASK 2: Matrix and Vector Operations
# TASK 2.1 Determinant
def determinant(matrix: list[list[float]]) -> float:
    return (matrix[0][0]*matrix[1][1]*matrix[2][2] + matrix[2][0]*matrix[0][1]*matrix[1][2] + matrix[0][2]*matrix[1][0]*matrix[2][1]
            - matrix[2][0]*matrix[1][1]*matrix[0][2] - matrix[0][0]*matrix[2][1]*matrix[1][2] - matrix[2][2]*matrix[1][0]*matrix[0][1])

#TASK 2.2 Trace
def trace(matrix: list[list[float]]) -> float:
    return matrix[0][0]+matrix[1][1]+matrix[2][2]

#TASK 2.3 Vector Norm
def norm(vector: list[float]) -> float:
    return math.sqrt(vector[0]*vector[0] + vector[1]*vector[1] + vector[2]*vector[2])

#TASK 2.4 Transpose of matrix
def transpose(matrix: list[list[float]]) -> list[list[float]]:
    T= [[],[],[]]
    for i in range(3):
        for j in range(3):
            T[i].append(matrix[j][i])

    return T

#TASK 2.5 Matrix-vector multiplication
def multiply(matrix: list[list[float]], vector: list[float]) -> list[float]:
    P = []
    for line in matrix:
        id=0
        suma=0
        for element in line:
            #print(element, vector[id])
            suma += element * vector[id]
            id+=1
        P.append(suma)

    return P




A, B = load_system(pathlib.Path("system.txt"))

print(f"{A=} {B=}")
print(f"{determinant(A)=}")
print(f"{trace(A)=}")
print(f"{norm(B)=}")
print(f"{transpose(A)=}")
print(f"{multiply(A, B)=}")

'''
A=[[2, 3, -1], [1, -1, 4], [3, 1, 2]] B=[5, 6, 7]
determinant(A)=14
trace(A)=3
norm(B)=10.488088481701515
transpose(A)=[[2, 1, 3], [3, -1, 1], [-1, 4, 2]]
multiply(A, B)=[21, 27, 35]
'''