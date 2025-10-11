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

# TASK 3: Solving using Cramer’s Rule
def solve_cramer(matrix: list[list[float]], vector: list[float]) -> list[float]:
    AX = copy.deepcopy(A)
    AY = copy.deepcopy(A)
    AZ = copy.deepcopy(A)
    '''
    Pentru a stii dimenisunea matricelor
    cu [[],[],[]] dadea index out of bounds
    AX=A.copy nu este suficient la lsite de liste
    '''
    for column in range(3):
        for line in range(3):
            if column==0: AX[line][column] = vector[line]
            if column==1: AY[line][column] = vector[line]
            if column==2: AZ[line][column] = vector[line]

    #print(AX, AY, AZ)

    x=determinant(AX)/determinant(A)
    y=determinant(AY)/determinant(A)
    z=determinant(AZ)/determinant(A)

    return [x, y, z]


# TASK 4: Solving using Cramer’s Rule
def minor(matrix: list[list[float]], i: int, j: int) -> list[list[float]]:
    M = [[],[]]
    el=0
    for line in range(3):
        for column in range(3):
            if line!=i and column!=j:
                if (el==0 or el==1): M[0].append(matrix[line][column])
                if (el==2 or el==3): M[1].append(matrix[line][column])
                el+=1

    return M

def cofactor(matrix: list[list[float]]) -> list[list[float]]:
    C=copy.deepcopy(matrix)
    for line in range(3):
        for column in range(3):
            Minor=minor(matrix,line,column)
            det_2x2=Minor[0][0]*Minor[1][1]-Minor[1][0]*Minor[0][1]
            C[line][column]=det_2x2
            if (line+column)%2==1: C[line][column]*=-1

    return C

def adjoint(matrix: list[list[float]]) -> list[list[float]]:
    return transpose(cofactor(matrix))


def solve(matrix: list[list[float]], vector: list[float]) -> list[float]:
    scalar= 1/determinant(matrix)
    inverse = copy.deepcopy(adjoint(matrix))

    for i in range(3):
        for j in range(3):
            inverse[i][j]*=scalar

    Result=multiply(inverse,vector)
    return Result




A, B = load_system(pathlib.Path("system.txt"))

print(f"{A=} {B=}")
print(f"{determinant(A)=}")
print(f"{trace(A)=}")
print(f"{norm(B)=}")
print(f"{transpose(A)=}")
print(f"{multiply(A, B)=}")
print(f"{solve_cramer(A, B)=}")
print(f"{solve(A, B)=}")
'''
A=[[2, 3, -1], [1, -1, 4], [3, 1, 2]] B=[5, 6, 7]
determinant(A)=14
trace(A)=3
norm(B)=10.488088481701515
transpose(A)=[[2, 1, 3], [3, -1, 1], [-1, 4, 2]]
multiply(A, B)=[21, 27, 35]
solve_cramer(A, B)=[0.35714285714285715, 2.0714285714285716, 1.9285714285714286]
solve(A, B)=[0.35714285714285765, 2.071428571428571, 1.9285714285714293]
'''