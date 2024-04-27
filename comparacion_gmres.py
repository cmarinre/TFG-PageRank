import numpy as np
import time
from gmres_reiniciado import GMRESReiniciado
from gmres import GMRES
from funciones_comunes import matrizPageRank, multiplicacionValorVector, multiplicacionValorMatriz



def comparacion_gmres(A):
    alpha = 0.85
    N = len(A)

    # Primero formateamos nuestro problema a la forma Ax=b
    # Nuestro sistema es de la forma (I-alphaA)x = (1-alpha)v

    # Nuestro vector b, que en nuestro caso es (1-alpha)v
    v = np.ones(N) / N
    b = multiplicacionValorVector(1-alpha, v)

    # Nuestra matriz, que es (I-alpha(A))
    Matriz = np.eye(N) - np.array(multiplicacionValorMatriz(alpha, A))

    # Necesitamos un vector inicial x_0
    x_0 = np.random.rand(N)
    x_0 = x_0 / np.linalg.norm(np.array(x_0), ord=1)


    start_time = time.time()
    x_n, iteraciones = GMRES(Matriz, b, x_0, 1000, 0.0000000000000001)
    end_time = time.time()
    elapsed_time = end_time - start_time

    print("El tiempo de ejecución de GMRES fue de: {:.5f} segundos".format(elapsed_time))
    # print("Vector solución", np.array(x_n))
    print("Número de iteraciones:", iteraciones)

    start_time_rei = time.time()
    x_n_rei, iteraciones_rei = GMRESReiniciado(Matriz, b, x_0, 0.0000000000000001, 3)
    end_time_rei = time.time()
    elapsed_time_rei = end_time_rei - start_time_rei

    print("El tiempo de ejecución de GMRES reiniciado fue de: {:.5f} segundos".format(elapsed_time_rei))
    # print("Vector solución", np.array(x_n_rei))
    print("Número de iteraciones:", iteraciones_rei)




if __name__ == "__main__":

    print("Creando matriz")
    A = matrizPageRank(5000)
    print("matriz creada")

    comparacion_gmres(A)