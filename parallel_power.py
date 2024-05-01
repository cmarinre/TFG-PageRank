import copy

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
    salir = False
    while k < max_iterations and salir==False:
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
            resta = [abs(x_k[i][j] - x_k_1[i][j]) for j in range(N)]
            if all(abs(valor) < tolerance for valor in resta):
                salir = True

        x_k_1 = copy.deepcopy(x_k)
        k+=1

    return x_k, k


def paraller_power_modified(A, vector, max_mv, tolerance, alphas):
    s = len(alphas)
    N = len(A)

    x = np.zeros((s, N))
    r = np.ones((s, N))
    res = np.ones(s)

    u = multiplicacionMatrizVector(A, v) - v
    
    for i in range(s):
        r[i] = alphas[i]*u
        res[i] = np.linalg.norm(r[i], ord=2)
        if res[i]>= tolerance:
            x[i] = r[i] + v

    mv=1

    while max(res) >= tolerance and mv <= max_mv:
        u = multiplicacionMatrizVector(A,v)
        mv += 1
        
        for i in range(s):
            if res[i]>=tolerance:
                r[i] = alphas[i]*u
                res[i] = np.linalg.norm(r[i], ord=2)

            if res[i]>= tolerance:
                x[i] = r[i] + x[i]

        for i in range(s):
            x[i] = x[i] / np.linalg.norm(x[i], ord=1)

    return x, mv



if __name__ == "__main__":
    A = np.array([[1/2, 1/3, 0, 0],
                [0, 1/3, 0, 1],
                [0, 1/3, 1/2, 0],
                [1/2, 0, 1/2, 0]])
    N = len(A)
    v = np.ones(N) 
    x_k, num_it = paraller_power(A, v, 10000, 0.000000001, [0.25, 0.33, 0.5, 0.66, 0.75, 0.85])

    print(x_k)
    print(num_it)

    x, num_it2 = paraller_power_modified(A, v, 10000, 0.000001, [0.25, 0.33, 0.5, 0.66, 0.75, 0.85])

    print(x)
    print(num_it2)
