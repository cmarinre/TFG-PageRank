
import time

import numpy as np

from funciones_comunes import (arreglarNodosColgantes, modificarMatriz,
                               obtenerSolucionPython, residuoDosVectores)
from gmres_reiniciado import GMRES_m
from read_data import read_data, read_data_minnesota


# Implementación del método POWER-GMRES(m)
def power_gmres(P, b, alpha, x, max_it, tol, alpha_1, m):
    A = np.eye(len(P)) - np.array(np.dot(alpha, P))
    terminado = False
    while terminado==False:     
        r=1
        for i in range(0,2):
            # Aplicación del método GMRES REINICIADO
            x, conver = GMRES_m(A, b, x, m, tol)

        if conver>tol:
            num_it=0
            while num_it < max_it and r>tol:
                x = x/np.linalg.norm(x, ord=1)
                z = np.dot(P, x)
                r = np.linalg.norm(np.dot(alpha, z) + b - x, ord=2)
                r_0 = r
                r_1 = r
                ratio = 0
                while ratio < alpha_1 and r > tol:
                    x = np.dot(alpha, z) + b
                    z = np.dot(P, x)
                    r = np.linalg.norm(np.dot(alpha, z) + b - x, ord=2)
                    ratio = r/r_0
                    r_0 = r
                x = np.dot(alpha, z) + b
                x = x/np.linalg.norm(x, ord=1)
                if(r/r_1 > alpha_1):
                    num_it = num_it+1

            if(r<tol):
                terminado = True
        else:
            terminado = True

    return x

if __name__ == "__main__":

    # P = read_data_minnesota("./datos/minnesota2642.mtx")
    P = read_data("./datos/hollins6012.mtx")
    # P = read_data("./datos/stanford9914.mtx")
    P = arreglarNodosColgantes(P)

    alpha = 0.95

    N = len(P)

    M = modificarMatriz(P, alpha)

    # Primero formateamos nuestro problema a la forma Ax=b
    # Nuestro sistema es de la forma (I-alphaP)x = (1-alpha)v PERO PASAMOS P y luego lo modificamos

    # Nuestro vector b, que en nuestro caso es (1-alpha)v
    v = np.ones(N) / N    
    b = np.dot(1-alpha, v)

    # Necesitamos un vector inicial x_0
    x_0 = np.ones(N)/N

    tol = 1e-8
    m = 2
    max_it=10

    start_time = time.time()
    x_n = power_gmres(P, b, alpha, x_0, max_it, tol, alpha-0.1, m)

    end_time = time.time()
    elapsed_time = end_time - start_time


    print("El tiempo de ejecución de POWER- GMRES fue de: {:.5f} segundos".format(elapsed_time))
    print("Vector solución normalizado", x_n)

    diferencia = residuoDosVectores(x_n, np.dot(M, x_n))
    print("diferencia", diferencia)


