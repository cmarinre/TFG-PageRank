import copy
import time

import numpy as np

from comparacion_gmres import comparacion_gmres
from comparacion_powers import comparacionPowersMult
from funciones_comunes import matrizPageRank, modificarMatriz
from power_gmres import power_gmres


def ejecucionPowerGmres(Matriz, M, b, x_0, max_it, tol, alpha_1, m):
    
    start_time = time.time()
    x_n = power_gmres(Matriz, M, b, x_0, max_it, tol, alpha_1, m)
    end_time = time.time()
    elapsed_time = end_time - start_time

    print("El tiempo de ejecución de GMRES fue de: {:.5f} segundos".format(elapsed_time))
    print("El vector propio es :", x_n)




if __name__ == "__main__":

    print("Creando matriz")
    A = matrizPageRank(50)
    # A = np.array([[1/2, 1/3, 0, 0],
    #             [0, 1/3, 0, 1],
    #             [0, 1/3, 1/2, 0],
    #             [1/2, 0, 1/2, 0]])
    M = modificarMatriz(A, 0.85)
    print("Matriz creada")

    alpha = 0.85
    N = len(A)

    # Primero formateamos nuestro problema a la forma Ax=b
    # Nuestro sistema es de la forma (I-alphaA)x = (1-alpha)v

    # Nuestro vector b, que en nuestro caso es (1-alpha)v
    v = np.ones(N) / N
    b = np.dot(1-alpha, v)

    # Nuestra matriz, que es (I-alpha(A))
    Matriz = np.eye(N) - np.array(np.dot(alpha, copy.deepcopy(A)))

    tol = 1e-20
    max_it = 10000
    m=3
    alpha_1 = tol
    N = len(A)

    # Necesitamos un vector inicial x_0
    x_0 = np.random.rand(N)
    x_0 = x_0 / np.linalg.norm(np.array(x_0), ord=1)

    print("Comienza Power")
    comparacionPowersMult(copy.deepcopy(M), x_0, max_it, tol)
    print("Termina Power")

    print("Comienza GMRES")
    comparacion_gmres(copy.deepcopy(Matriz), b, x_0, max_it, tol, m)
    print("Termina GMRES")

    print("Comienza Power GMRES")
    ejecucionPowerGmres(copy.deepcopy(Matriz), M, b, x_0, max_it, tol, alpha_1, m)
    print("Termina Power GMRES")

