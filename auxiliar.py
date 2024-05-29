import copy
import time

import numpy as np

from funciones_comunes import (arreglarNodosColgantes,
                               obtenerComparacionesNumpySoluciones,
                               obtenerSolucionesNumpy)
from read_data import read_data, read_data_cz1268


def arnoldi_givens(A, b, x_0, max_it, alpha_k, mv):

    N = len(A)
    
    # Y nuestro vector r_0, b-Ax_0
    r_0 = b - np.dot(A, x_0)


    # Generamos una matriz V y una h con todos sus valores a 0
    V = np.zeros((N, max_it+1))
    h = np.zeros((max_it+1, max_it))
    
    # Establecemos el v_1 al vector inicial normalizado.
    r_0_norm = np.linalg.norm(r_0, ord=2)
    V[:, 0] = np.array(r_0 / r_0_norm)
    
    # Inicializamos el vector g
    g = np.zeros(max_it+1)
    g[0] = r_0_norm
    
    betae1 =  np.zeros(N)
    betae1[0] = r_0_norm.copy()

    # Vector solucion
    x = np.zeros(N)
    
    # Como trabajamos con matrices a las que accedemos desde el 0, reducimos 1 el número N  y num_cols
    N = N-1
    
    n=0
    while n<=(max_it-1):

        t = np.dot(A, V[:,n])
        
        # Arnoldi
        for i in range(0, n+1):           
            h[i][n] = np.dot(V[:,i], t)                      
            t = t-np.dot(h[i][n], V[:,i])
            i+=1
        t_norm = np.linalg.norm(t, ord=2)
        h[n+1][n] = t_norm
        V[:,n+1] = t / t_norm
        
        # Givens
        for j in range(0, n):      
            c_j = abs(h[j][j]) / (np.sqrt( h[j][j]**2 + h[j+1][j]**2  ))
            s_j = (h[j+1][j] / h[j][j])*c_j

            temp = c_j*h[j][n] + s_j*h[j+1][n]
            h[j+1][n] = -s_j*h[j][n] + c_j*h[j+1][n]
            h[j][n] = temp
                     

        delta = np.sqrt(h[n, n] ** 2 + h[n + 1, n] ** 2)
        c_n = h[n, n] / delta
        s_n = h[n + 1, n] / delta

        h[n][n] = c_n*h[n][n] + s_n*h[n+1][n]
        h[n+1][n] = 0
        
        g[n+1] = -s_n*g[n]
        g[n] = c_n*g[n]
        
        if n==(max_it-1):
            y = np.linalg.solve(h[:(n), :(n)], g[:(n)])
            aux = np.dot(V[:, :(n)], y)
            x = x_0 + aux
            x = x/np.linalg.norm(x, ord=1)


            res = alpha_k*(np.linalg.norm(betae1 - aux, ord=2))
            mv = mv + n

        #Aumentamos la iteración
        n +=1

    return x, res, h[:(n), :(n)], y, V[:, :(n)], mv, r_0_norm




def  parallel_gmres(P, x_0, max_it, tol, alphas, m):
    # Calculamos la dimension de P y cuántos alphas tenemos
    N=len(P)
    s=len(alphas)

    v = np.ones(N) / N

    iter = 1
    betae1 =  np.zeros(m)

    res = np.zeros(s)
    x = np.zeros((s, N))
    num_it = np.zeros(s)
    r_0 = np.zeros((s, N))

    for i in range(s):
        # Para no acceder al vector varias veces guardamos el valor de alpha_i
        alpha_i  = alphas[i]
        r_0[i] = ((1-alpha_i)/alpha_i)*v - np.dot(np.dot((1/alpha_i), np.eye(N)) - P, x_0[i])
        res[i] = alpha_i*np.linalg.norm(r_0[i], ord=2)

    mv = 0


    y = np.zeros((s, 2))

    h = [np.zeros((m, m)) for _ in range(s)]

    gamma = np.zeros(s)
    while max(res) >= tol and iter <= max_it:
        k = np.argmax(res)
        alpha_k = alphas[k]
        for i in range(s):
            if(i!=k): gamma[i] = (res[i]*alpha_k)/(res[k]*alphas[i])

        x[k], res[k], h[k], y[k], V, mv, beta = arnoldi_givens(np.eye(N)/alpha_k-P, ((1-alpha_k)/alpha_k)*v , x_0[k], m, alpha_k, mv)
        
        if res[k]<tol: num_it[k] = mv

        for i in range(s):
            if i != k:
                if res[i]>=tol:
                    alpha_i  = alphas[i]
                    h[i] = h[k] + ( (1-alpha_i)/alpha_i - (1-alpha_k)/alpha_k )*np.eye(len(h[k]))

                    betae1[0] = beta
                    z = betae1 - np.dot(h[k], y[k])
                    b = gamma[i] * betae1
                    A = np.column_stack((h[i], z))

                    solution = np.linalg.lstsq(A, b, rcond=None)[0]

                    # Extraemos y^i y gamma^i de la solución
                    y_i = solution[:-1]
                    gamma[i] = solution[-1]  # El último elemento

                    x[i] = x_0[i] + np.dot(V, y_i)
                    x[i] = x[i]/np.linalg.norm(x[i], ord=1)
                    # res[i] = (alpha_i/alpha_k)*gamma[i]*res[k]
                    res[i] = alphas[k]*np.linalg.norm(((1-alpha_i)/alpha_i)*v - np.dot(np.dot((1/alpha_i), np.eye(N)) - P, x[i]), ord=2)
                else:
                    # Si ya ha cumplido el criterio de convergencia
                    # Y no hemos guardado antes su núm it (está a 0) , guardamos el número de iteraciones
                    if num_it[i]==0: num_it[i] = mv
        x_0 = x.copy()
        iter += 1

    return x, num_it, res



if __name__ == "__main__":
    # P = read_data_cz1268("./datos/cz1268.mtx")
    # P = read_data("./datos/minnesota2642.mtx")
    # # P = read_data("./datos/hollins6012.mtx")
    # # P = read_data("./datos/stanford9914.mtx")
    # P = arreglarNodosColgantes(P)

    P = np.array([[1/2, 1/3, 0, 0],
                  [0, 1/3, 0, 1],
                  [0, 1/3, 1/2, 0],
                  [1/2, 0, 1/2, 0]])
    
    N = len(P)
    tol=1e-4
    max_it = 1000
    m = 2

    alphas = np.zeros(50)
    for i in range(50):
        alphas[i] = (i+50)*0.01
    # alphas = np.array([0.85, 0.9, 0.99])
    print(alphas)

    # x_0 = np.random.rand(len(P))
    # x_0 = x_0 / np.linalg.norm(np.array(x_0), ord=1)
    x_0 = np.ones(N)/N

    start_time = time.time()
    x, num_it, res = parallel_gmres(P, np.tile(x_0, (len(alphas), 1)), max_it, tol, alphas, m)
    end_time = time.time()
    elapsed_time = end_time - start_time

    print("El tiempo de ejecución del parallel gmres fue de: {:.5f} segundos".format(elapsed_time))
    print("Número de iteraciones", num_it)

    soluciones = obtenerSolucionesNumpy(P, alphas)
    normas = obtenerComparacionesNumpySoluciones(x, soluciones)
    
    print("Vectores solución", x)
    print("Residuo", res)
    print("Normas residuales", normas)