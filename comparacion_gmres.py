import numpy as np
import time
from gmres_reiniciado import GMRESReiniciado
from gmres import GMRES
from funciones_comunes import matrizPageRank



def comparacion_gmres(Matriz, b, x_0, max_it, tol, m):
    
    start_time = time.time()
    x_n, iteraciones = GMRES(Matriz, b, x_0, max_it, tol)
    end_time = time.time()
    elapsed_time = end_time - start_time

    print("El tiempo de ejecución de GMRES fue de: {:.5f} segundos".format(elapsed_time))
    print("Vector solución", np.array(x_n))
    print("Número de iteraciones:", iteraciones)

    start_time_rei = time.time()
    x_n_rei, iteraciones_rei = GMRESReiniciado(Matriz, b, x_0, tol, m, max_it)
    end_time_rei = time.time()
    elapsed_time_rei = end_time_rei - start_time_rei

    print("El tiempo de ejecución de GMRES reiniciado fue de: {:.5f} segundos".format(elapsed_time_rei))
    print("Vector solución", np.array(x_n_rei))
    print("Número de iteraciones:", iteraciones_rei)




if __name__ == "__main__":

    print("Creando matriz")
    A = matrizPageRank(5000)
    print("matriz creada")

    comparacion_gmres(A, 5000, 0.000000001, 3)