import copy
import time

import numpy as np

from funciones_comunes import arreglarNodosColgantes, modificarMatriz
from power_method import power_method, power_method_convergence
from power_method_adaptive import (adaptive_power_method,
                                   adaptive_power_method_k)
from read_data import read_data


# Función que ejecuta el power adaptive k
def ejecucionPowersAdaptiveK(A, max_it, tol, k):

    # Registro del tiempo de inicio
    start_time1 = time.time()
    # Aplicación del método de las potencias
    eigenvector1, num_it1 = adaptive_power_method_k(A, x_0, max_it, tol, k)
    # Registro del tiempo de finalización
    end_time1 = time.time()
    # Cálculo del tiempo transcurrido
    elapsed_time1 = end_time1 - start_time1

    print("El tiempo de ejecución de power adpative k=8 fue de: {:.5f} segundos".format(elapsed_time1))
    print("Número de iteraciones:", num_it1)
    # print("Suma de los valores del vector propio para asegurar que su norma 1 es igual a 1:", np.sum(eigenvector1))
    print("Vector propio:", np.array(eigenvector1[0]))



# Función que compara los métodos de potencias y potencias adaptado, mirando la convergencia del primero
# como la convergencia de sus componentes.
def comparacionPowersMult(A, x_0, max_it, tol):

    # Registro del tiempo de inicio
    start_time2 = time.time()
    # Aplicación del método de las potencias adaptado
    eigenvector2, num_it2 = adaptive_power_method(copy.deepcopy(A), copy.deepcopy(x_0), max_it, tol)
    # Registro del tiempo de finalización
    end_time2 = time.time()
    # Cálculo del tiempo transcurrido
    elapsed_time2 = end_time2 - start_time2

    print("El tiempo de ejecución de adaptive fue de: {:.5f} segundos".format(elapsed_time2))
    print("Número de iteraciones:", num_it2)
    # print("Suma de los valores del vector propio para asegurar que su norma 1 es igual a 1:", np.sum(eigenvector2))
    print("Vector propio:", np.array(eigenvector2[0]))



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
    print("Vector propio:", np.array(eigenvector1[0]))


# Función que ejecuta el método de las potencias estandar
def ejecucionPowerEstandar(A, x_0, max_it, tol):

    # Registro del tiempo de inicio
    start_time1 = time.time()
    # Aplicación del método de las potencias
    eigenvector1, num_it1 = power_method(A, x_0, max_it, tol)
    # Registro del tiempo de finalización
    end_time1 = time.time()
    # Cálculo del tiempo transcurrido
    elapsed_time1 = end_time1 - start_time1

    print("El tiempo de ejecución de power fue de: {:.5f} segundos".format(elapsed_time1))
    print("Número de iteraciones:", num_it1)
    # print("Suma de los valores del vector propio para asegurar que su norma 1 es igual a 1:", np.sum(eigenvector1))
    print("Vector propio:", np.array(eigenvector1[0]))
    return eigenvector1, num_it1



if __name__ == "__main__":

    A = read_data("./datos/minnesota2642.mtx")
    A = arreglarNodosColgantes(A)

    print("Modificando matriz")
    M = modificarMatriz(A, 0.85)
    print("Matriz modificada")

    N = len(A)
    x_0 = np.random.rand(N)
    x_0 = x_0 / np.linalg.norm(np.array(x_0), ord=1)
    
    comparacionPowersMult(copy.deepcopy(M), x_0, 5000, 0.00000001)

    ejecucionPowersAdaptiveK(copy.deepcopy(M), x_0, 5000, 0.00000001, 8)

    ejecucionPowerEstandar(copy.deepcopy(M), x_0, 5000, 0.00000001)

    
