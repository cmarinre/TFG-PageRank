import copy
import time

import numpy as np

from funciones_comunes import (arreglarNodosColgantes,
                               obtenerComparacionesNumpySoluciones,
                               obtenerSolucionesNumpy)
from read_data import read_data, read_data_cz1268


def arnoldi_givens(A, r_0, m, x_0, alpha):
    N = len(A)

    V = np.zeros((N, m+1))
    h = np.zeros((m+1, m))
 

    r_0_norm = np.linalg.norm(r_0, ord=2)

    betae1 = np.zeros(N)
    betae1[0] = r_0_norm.copy()

    g = np.zeros(m+1)
    g[0] = r_0_norm.copy()

    V[:, 0] = r_0 / r_0_norm


    n=0
    while n<=(m-1):
        t = np.dot(A, V[:,n])
        # Arnoldi
        i=0
        for i in range(0, n+1):           
            h[i,n] = np.dot(V[:,i], t)                      
            t -= np.dot(h[i,n], V[:,i])

        t_norm = np.linalg.norm(t, ord=2)
        h[n+1,n] = t_norm
        V[:,n+1] = t / t_norm
        
        # Givens
        for j in range(0, n):
            c_j = abs(h[j][j]) / (np.sqrt( h[j][j]**2 + h[j+1][j]**2  ))
            s_j = (h[j+1][j] / h[j][j])*c_j

            apply_givens_rotation(h, c_j, s_j, j, n)

            
        delta = np.sqrt(h[n, n] ** 2 + h[n + 1, n] ** 2)
        c_n = h[n, n] / delta
        s_n = h[n + 1, n] / delta

        h[n][n] = c_n*h[n][n] + s_n*h[n+1][n]
        h[n+1][n] = 0
        
        g[n+1] = -s_n*g[n]
        g[n] = c_n*g[n]
        
        n +=1


    # Al acabar con GIVENS y ARNOLDI, calculamos la x:
    y = np.linalg.solve(h[:(m), :(m)], g[:(m)])
    x = x_0 + np.dot(V[:, :(m)], y)


    res = alpha*np.linalg.norm(betae1[:m] - np.dot(h[:(m), :(m)], y))
    return x, y, res, h, V, betae1, n
    

def apply_givens_rotation(h, c, s, k, i):
    temp = c * h[k, i] + s * h[k + 1, i]
    h[k + 1, i] = -s * h[k, i] + c * h[k + 1, i]
    h[k, i] = temp




def parallel_gmres(P, v, m, alphas, tol, max_it, x_0):
    N = len(P)
    s = len(alphas)

    r_0 = np.zeros((s, N))
    res = np.zeros(s)
    gamma = np.zeros(s)
    x = x_0.copy()
    h = [np.zeros((m+1, m)) for _ in range(s)]

    for i in range(s):
        alpha_i = alphas[i]
        r_0[i] = np.dot((1-alpha_i)/alpha_i, v) - np.dot(np.eye(N)/alpha_i - P, x_0[i])
        res[i] = np.dot(alpha_i, np.linalg.norm(r_0[i], ord=2))
    mv = 0
    iter = 1
    numit = np.zeros(s)

    while max(res)>=tol and iter <= max_it :

        k = np.argmax(res)
        for i in range(s):
            if i!=k: 
                gamma[i] = res[i]*alphas[k]/(res[k]*alphas[i])
        r_0[k] = np.dot((1-alphas[k])/alphas[k], v) - np.dot(np.eye(N)/alphas[k] - P, x_0[k])
        x[k], y_k, res[k], h[k], V, betae1, suma = arnoldi_givens(np.eye(N)/alphas[k]-P, r_0[k], m, x_0[k], alphas[k]) 
        mv = mv+suma
        if res[k]<tol: numit[k] = mv

        for i in range(s):
            if i!=k:
                if res[i] >= tol:
                    
                    h[i] = h[k][:(m),:(m)] + np.dot( (1-alphas[i])/alphas[i] - (1-alphas[k])/alphas[k] , np.eye(m))
                    
                    z = betae1[:(m)] - np.dot(h[k][:(m),:(m)], y_k)
                    A = np.column_stack((h[i][:(m),:(m)], z))
                    b = np.dot(gamma[i], betae1[:(m)])
                    solution = np.linalg.lstsq(A, b, rcond=None)[0]

                    y_i = solution[:-1] # Extraemos y^i y gamma^i de la solución
                    gamma[i] = solution[-1]  # El último elemento

                    x[i] = x_0[i] + np.dot(V[:,:(m)], y_i)
                    x[i] = x[i] / np.linalg.norm(x[i], ord=1)
                    # res[i] = alphas[i]/alphas[k]*gamma[i]*res[k]
                    res[i] = alphas[k]*np.linalg.norm((1 - alphas[i]) / alphas[i] * v - (np.eye(N) / alphas[i] - P).dot(x[i]), ord=2) # CON LA QUE NOS FUNCIONA XD
                else:
                    if(numit[i]==0): numit[i] = mv

        x_0 = x.copy()

        iter += 1
    
    return x_0, res, mv, numit

if __name__ == "__main__":

    P = read_data_cz1268("./datos/cz1268.mtx")
    # P = read_data("./datos/minnesota2642.mtx")
    # P = read_data("./datos/hollins6012.mtx")
    # P = read_data("./datos/stanford9914.mtx")
    P = arreglarNodosColgantes(P)

    # P = np.array([[1/2, 1/3, 0, 0],
    #         [0, 1/3, 0, 1],
    #         [0, 1/3, 1/2, 0],
    #         [1/2, 0, 1/2, 0]])

    N = len(P)

    v = np.ones(N)/N
    m = 2
    alphas = np.zeros(50)
    for i in range(0,50):
        alphas[i] = (i+50)*0.01
    # alphas= np.array([0.99])
    tol = 1e-4
    max_it = 100000
    x_0_1 = np.zeros(N)

    x_0 = np.tile(x_0_1, (len(alphas), 1))

    
    start_time = time.time()
    x, res, mv, numit = parallel_gmres(P, v, m, alphas, tol, max_it, x_0)
    end_time = time.time()
    elapsed_time = end_time - start_time

    print("El tiempo de ejecución del PARALLEL gmres fue de: {:.5f} segundos".format(elapsed_time))

    soluciones = obtenerSolucionesNumpy(P, alphas)
    normas = obtenerComparacionesNumpySoluciones(x, soluciones)
    
    # print("Solucion python", soluciones)
    # print("Vectores solución", x)

    print("Número de iteraciones", mv)
    print("Número de iteraciones", numit)

    print("Residuo", res)
    print("Norma residual", normas)



