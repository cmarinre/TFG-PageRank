import numpy as np

from funciones_comunes import multiplicacionMatrizVector


def paraller_power(A, vector, max_iterations, tolerance, alphas):
    k=0
    u1=[]
    u2=[]
    s = len(alphas)
    N = len(A)
    x_k = np.zeros((s, N))
    x_k_1 = np.zeros((s, N))
    while k < max_iterations:
        if k==0:
            u1 = multiplicacionMatrizVector(A, vector)
            u2 = vector
            for i in range(s):
                x_k[i] = np.dot(alphas[i], u1) + np.dot((1-alphas[i]), u2)
        else:
            for i in range(s):
                x_k[i] = x_k_1[i] - np.dot(alphas[i], u1)
            
            u1 = multiplicacionMatrizVector(A, u1)
            u2 = multiplicacionMatrizVector(A, u2)

            for i in range(s):
                x_k[i] = np.dot(alphas[i], u1) + x_k[i] + np.dot((1-alphas[i])*alphas[i], u2)
        
        for i in range(s):
            x_k[i] = x_k[i] / np.linalg.norm(x_k[i], ord=1)

        for i in range(s):
            print(i)
            all_converg = 0
            # print(x_k[i])
            # print(x_k[i][0])
            # print(x_k_1[i])
            resta = [abs(x_k[i][j] - x_k_1[i][j]) for j in range(N)]
            if all(abs(valor) < tolerance for valor in resta):
                all_converg+=1

        if(all_converg==s):
            break

        x_k_1 = x_k
        k+=1

    return x_k

if __name__ == "__main__":
    A = np.array([[1/2, 1/3, 0, 0],
                [0, 1/3, 0, 1],
                [0, 1/3, 1/2, 0],
                [1/2, 0, 1/2, 0]])
    N = len(A)
    v = np.ones(N) 
    x_k = paraller_power(A, v, 10, 0.000000001, [0.25, 0.33, 0.5, 0.66, 0.75,1])
    print(x_k)
