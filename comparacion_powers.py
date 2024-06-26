import copy
import time

import numpy as np

from funciones_comunes import arreglarNodosColgantes, modificarMatriz
from power_method import power_method, power_method_convergence
from power_method_adaptive import (adaptive_power_method,
                                   adaptive_power_method_k)
from read_data import read_data, read_data_minnesota


# Función que ejecuta el power adaptive k
def ejecucionPowersAdaptiveK(A, x_0, max_it, tol, k):

    # Registro del tiempo de inicio
    start_time1 = time.time()
    # Aplicación del método de las potencias
    eigenvector1, num_it1 = adaptive_power_method_k(A, x_0, max_it, tol, k)
    # Registro del tiempo de finalización
    end_time1 = time.time()
    # Cálculo del tiempo transcurrido
    elapsed_time1 = end_time1 - start_time1

    print("El tiempo de ejecución de power ADAPTIVE PART k=8 fue de: {:.5f} segundos".format(elapsed_time1))
    print("Número de iteraciones:", num_it1)
    print("Vector propio:", np.array(eigenvector1))



# Función que compara los métodos de potencias y potencias adaptado, mirando la convergencia del primero
# como la convergencia de sus componentes.
def comparacionPowersMult(A, x_0, max_it, tol):

    A_copy = copy.deepcopy(A)
    x_0_copy = copy.deepcopy(x_0)

    # Registro del tiempo de inicio
    start_time1 = time.time()
    # Aplicación del método de las potencias
    eigenvector3, num_it1 = power_method_convergence(A, x_0, max_it, tol)
    # Registro del tiempo de finalización
    end_time1 = time.time()
    # Cálculo del tiempo transcurrido
    elapsed_time1 = end_time1 - start_time1

    print("El tiempo de ejecución de POWER PART CON NORMADIF fue de: {:.5f} segundos".format(elapsed_time1))
    print("Número de iteraciones:", num_it1)
    print("Vector propio:", np.array(eigenvector3))
    

    # Registro del tiempo de inicio
    start_time2 = time.time()
    # Aplicación del método de las potencias adaptado
    eigenvector2, num_it2 = adaptive_power_method(A_copy, x_0_copy, max_it, tol)
    # Registro del tiempo de finalización
    end_time2 = time.time()
    # Cálculo del tiempo transcurrido
    elapsed_time2 = end_time2 - start_time2

    print("El tiempo de ejecución de ADAPTIVE PART fue de: {:.5f} segundos".format(elapsed_time2))
    print("Número de iteraciones:", num_it2)
    print("Vector propio:", np.array(eigenvector2))

# Función que ejecuta el método de las potencias estandar y mide el tiempo
def ejecucionPowerEstandar(A, x_0, max_it, tol):

    # Registro del tiempo de inicio
    start_time1 = time.time()
    # Aplicación del método de las potencias
    eigenvector1, num_it1 = power_method(A, x_0, max_it, tol)
    # Registro del tiempo de finalización
    end_time1 = time.time()
    # Cálculo del tiempo transcurrido
    elapsed_time1 = end_time1 - start_time1

    print("El tiempo de ejecución de POWER NORMAL fue de: {:.5f} segundos".format(elapsed_time1))
    print("Número de iteraciones:", num_it1)
    # print("Vector propio:", np.array(eigenvector1))
    return eigenvector1, num_it1



if __name__ == "__main__":

    # P = read_data_minnesota("./datos/minnesota2642.mtx")
    # P = read_data("./datos/hollins6012.mtx")
    P = read_data("./datos/stanford9914.mtx")
    P = arreglarNodosColgantes(P)

    alpha = 0.85

    M = modificarMatriz(P, alpha)


    N = len(M)

    x_0 = np.ones(N)/N

    # tol = 1e-4

    #Descomentamos los que necesitemos en cada momento
    
    # comparacionPowersMult(copy.deepcopy(M), copy.deepcopy(x_0), 5000, tol)

    # ejecucionPowersAdaptiveK(copy.deepcopy(M), copy.deepcopy(x_0), 5000, tol, 8)

    # ejecucionPowerEstandar(copy.deepcopy(M), copy.deepcopy(x_0), 5000, tol)


    # tol = 1e-6

    # comparacionPowersMult(copy.deepcopy(M), copy.deepcopy(x_0), 5000, tol)

    # ejecucionPowersAdaptiveK(copy.deepcopy(M), copy.deepcopy(x_0), 5000, tol, 8)

    # ejecucionPowerEstandar(copy.deepcopy(M), copy.deepcopy(x_0), 5000, tol)


    tol = 1e-8

    # comparacionPowersMult(copy.deepcopy(M), copy.deepcopy(x_0), 5000, tol)

    ejecucionPowersAdaptiveK(copy.deepcopy(M), copy.deepcopy(x_0), 5000, tol, 8)

    # ejecucionPowerEstandar(copy.deepcopy(M), copy.deepcopy(x_0), 5000, tol)




    
