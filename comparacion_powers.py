import time
import numpy as np

from power_method import power_method, power_method_convergence
from power_method_adaptive import adaptive_power_method
from funciones_comunes import matrizPageRank, modificarMatriz


# Función que compara los métodos de potencias y potencias adaptado, mirando la convergencia del primero
# como la convergencia de la norma.
def comparacionPowers(A, max_it, tol):

    # Registro del tiempo de inicio
    start_time1 = time.time()
    # Aplicación del método de las potencias
    eigenvector1, num_it1 = power_method(A, max_it, tol)
    # Registro del tiempo de finalización
    end_time1 = time.time()
    # Cálculo del tiempo transcurrido
    elapsed_time1 = end_time1 - start_time1

    print("El tiempo de ejecución de power fue de: {:.5f} segundos".format(elapsed_time1))
    print("Número de iteraciones:", num_it1)
    print("Suma de los valores del vector propio para asegurar que su norma 1 es igual a 1:", np.sum(eigenvector1))
    print("Vector propio:", eigenvector1[0])

    # Registro del tiempo de inicio
    start_time2 = time.time()

    # Aplicación del método de las potencias adaptado
    eigenvector2, num_it2 = adaptive_power_method(A, max_it, tol)
    # Registro del tiempo de finalización
    end_time2 = time.time()
    # Cálculo del tiempo transcurrido
    elapsed_time2 = end_time2 - start_time2

    print("El tiempo de ejecución de adaptive fue de: {:.5f} segundos".format(elapsed_time2))
    print("Número de iteraciones:", num_it2)
    print("Suma de los valores del vector propio para asegurar que su norma 1 es igual a 1:", np.sum(eigenvector2))
    print("Vector propio:", eigenvector2[0])



# Función que compara los métodos de potencias y potencias adaptado, mirando la convergencia del primero
# como la convergencia de sus componentes.
def comparacionPowersMult(A, x_0, max_it, tol):

    # Registro del tiempo de inicio
    start_time1 = time.time()
    # Aplicación del método de las potencias
    eigenvector1, num_it1 = power_method_convergence(A, x_0, max_it, tol)
    # Registro del tiempo de finalización
    end_time1 = time.time()
    # Cálculo del tiempo transcurrido
    elapsed_time1 = end_time1 - start_time1

    print("El tiempo de ejecución de power fue de: {:.5f} segundos".format(elapsed_time1))
    print("Número de iteraciones:", num_it1)
    # print("Suma de los valores del vector propio para asegurar que su norma 1 es igual a 1:", np.sum(eigenvector1))
    print("Vector propio:", np.array(eigenvector1))

    # Registro del tiempo de inicio
    start_time2 = time.time()
    # Aplicación del método de las potencias adaptado
    eigenvector2, num_it2 = adaptive_power_method(A, x_0, max_it, tol)
    # Registro del tiempo de finalización
    end_time2 = time.time()
    # Cálculo del tiempo transcurrido
    elapsed_time2 = end_time2 - start_time2

    print("El tiempo de ejecución de adaptive fue de: {:.5f} segundos".format(elapsed_time2))
    print("Número de iteraciones:", num_it2)
    # print("Suma de los valores del vector propio para asegurar que su norma 1 es igual a 1:", np.sum(eigenvector2))
    print("Vector propio:", np.array(eigenvector2))





if __name__ == "__main__":
    # A = np.array([[1/2, 1/3, 0, 0],
    #               [0, 1/3, 0, 1],
    #               [0, 1/3, 1/2, 0],
    #               [1/2, 0, 1/2, 0]])

    # A = np.array([[0, 1/2, 1/3],
    #         [1/2, 0, 1/3],
    #         [1/2, 1/2, 1/3]])

    print("Creando matriz")
    A = matrizPageRank(300)
    print("Matriz creada")

    print("Modificando matriz")
    M = modificarMatriz(A, 0.85)
    print("Matriz modificada")

    comparacionPowersMult(M, 5000, 0.00000000001)
    # comparacionPowers(A)
    

