"""
Формируется матрица F следующим образом: скопировать в нее А и
если в Е максимальный элемент в нечетных столбцах больше, чем сумма чисел в нечетных строках,
то поменять местами С и В симметрично, иначе
В и Е поменять местами несимметрично.
При этом матрица А не меняется.
После чего если определитель матрицы А больше суммы диагональных элементов матрицы F, то
вычисляется выражение: A-1*AT – K * F-1, иначе
вычисляется выражение (AТ +G-FТ)*K, где G-нижняя треугольная матрица, полученная из А.
Выводятся по мере формирования А, F и все матричные операции последовательно.
"""
import time
import numpy as np
N = int(input("Введите количество строк (столбцов) квадратной матрицы больше 3 : "))
while N < 4 :
    N = int(input("Вы ввели неверное число\nВведите количество строк (столбцов) квадратной матрицы больше 3 :"))
K = int(input("Введите число К="))
start = time.time()
A = np.zeros((N,N),dtype=int)
F = np.zeros((N, N), dtype=int)
for i in range (N):
        for j in range(N):
            #A[i][j]=np.random.randint(-10,10)
            A[i][j] = i*10+j
middle = time.time()
print("A time = ", middle - start,"\nmatrix A: \n",A)
for i in range (N):
        for j in range(N):
            F[i][j]=A[i][j]
n = N // 2
E = np.zeros((n,n),dtype=int)
for i in range (n):
        for j in range (n):
            E[i][j]=A[i][j]
print ("E time = ", 0, "\n matrix E:\n",E)
enumerationL = []
enumerationC = []
for i in range(n):
     for j in range(n):
            if i%2==0 :
                enumerationC.append(E[i][j])
for i in range (n):
        for j in range(n):
            if j%2==0:
                enumerationL.append(E[i][j])
maxima=max(enumerationL)
summ = sum(enumerationC)
print(enumerationL, "\nMaximum element in columns= ",maxima , "\n",enumerationC, "\nSum element in lines = ",summ )
if summ < maxima:
        print("Так как Сумма элементов < максимального элемента => меняем B и С симметрично")
        for i in  range (n):
            for j in range (n):
                F[i][n+j]=A[N-1-i][n+j]
                F[N-1-i][n+j]=A[i][n+j]
else:
        print("Так как Сумма элементов > максимального элемента => меняем B и E несимметрично")
        F[0:n ,0:n] = A[0:n ,n+N%2 : N]
        F[0:n ,n+N%2 : N] = A[0:n ,0:n]
middle2 = time.time()
x = []
for i in range(N):
    for j in range(N):
        x.append(A[i][j])
c = min(x)
print("matrix A: \n",A , "\n matrix F: \n", F)
if np.linalg.det(A) == 0 or np.linalg.det(F) == 0:
        print("Матрица A или F вырождена => нельзя вычислить")
elif np.linalg.det(A) > sum(F.diagonal()):
        A = (A - np.dot(1,np.transpose(A)) - np.dot(K,(F-1)))
        finish = time.time()
        print("np.linalg.det(A) > sum(F.diagonal())","\nA time = :", finish - middle2,"\n",A)
else:
        A = np.dot(np.transpose(A) + np.tril(A) - np.transpose(F) ,K)
        finish = time.time()
        print("np.linalg.det(A) < sum(F.diagonal())\n", "\nТреугольная матрица из А :\n",np.tril(A) ,"\nA time = :", finish - middle2, "\n",A,"\n")
for i in A:  # делаем перебор всех строк матрицы
        for j in i:  # перебираем все элементы в строке
            print("%5d" % j, end=' ')
        print()
#######################################
from matplotlib import pyplot as plt
plt.title("Plot")
plt.xlabel("Numbers")
plt.ylabel("volumes")
for j in range (N):
    plt.plot([i for i in range(N)], A[j][::], marker='x')
plt.show()
#######################################
plt.title("Scatter")
plt.xlabel("Numbers")
plt.ylabel("volumes")
for j in range (N):
    plt.scatter([i for i in range(N)] , A[j][::])
plt.show()
#######################################
plt.title("Bar")
plt.xlabel("Numbers")
plt.ylabel("volumes")
for i in range (N):
 plt.bar([i for i in range(N)],A[::-1][i],width=1)
plt.show()
#######################################
print(c) #min element ? if c < 0 => pie can't exist
if c >= 0:
       print("min elemet in A, >= 0 ")
       for i in range (N):
         plt.pie(A[i][::])
else :
    print("min element in A, <0 , pie can't exist")
plt.show()
#######################################
OX=np.zeros((N),dtype=int)
for i in range (N):
    OX[i] = i
C = plt.contourf(OX,OX,A,8,
                  colors = 'black')
plt.contourf(OX,OX,A,10,cmap=plt.cm.summer)
plt.clabel(C,inline = 1 , fontsize = 8)
plt.colorbar()
plt.show()



