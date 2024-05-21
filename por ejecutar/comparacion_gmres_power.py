import copy
import time

import numpy as np

from funciones_comunes import arreglarNodosColgantes
from power_gmres import power_gmres, power_gmres2
from power_method import power_method
from read_data import read_data


def ejecucionPowerGmres(P, alpha, x_0, max_it, tol, alpha_1, m):
    
    N = len(P)

    # Nuestro vector b, que en nuestro caso es (1-alpha)v
    v = np.ones(N) / N    
    b = np.dot(1-alpha, v)

    start_time = time.time()
    x_n = power_gmres(P, b, alpha, x_0, max_it, tol, alpha_1, m)
    end_time = time.time()
    elapsed_time = end_time - start_time

    print("El tiempo de ejecución de GMRES-POWER fue de: {:.5f} segundos".format(elapsed_time))
    # print("El vector propio es :", x_n)
    return x_n

def ejecucionPowerGmres2(P, alpha, x_0, max_it, tol, alpha_1, m):
    
    N = len(P)

    # Nuestro vector b, que en nuestro caso es (1-alpha)v
    v = np.ones(N) / N    
    b = np.dot(1-alpha, v)

    start_time = time.time()
    x_n = power_gmres2(P, b, alpha, x_0, max_it, tol, alpha_1, m)
    end_time = time.time()
    elapsed_time = end_time - start_time

    print("El tiempo de ejecución de GMRES-POWER fue de: {:.5f} segundos".format(elapsed_time))
    # print("El vector propio es :", x_n)
    return x_n

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
    # print("Vector propio:", np.array(eigenvector1))
    return eigenvector1, num_it1


if __name__ == "__main__":

    # P = np.array([[1/2, 1/3, 0, 0],
    #               [0, 1/3, 0, 1],
    #               [0, 1/3, 1/2, 0],
    #               [1/2, 0, 1/2, 0]])


    # P = read_data("./datos/minnesota2642.mtx")
    P = read_data("./datos/hollins6012.mtx")
    # P = read_data("./datos/stanford9914.mtx")
    P = arreglarNodosColgantes(P)




    alpha = 0.85
    m = 5
    max_it = 20
    
    
    tol = 1e-4

    # Necesitamos un vector inicial x_0
    x_0 = np.random.rand(len(P))
    x_0 = x_0 / np.linalg.norm(np.array(x_0), ord=1)
    x_n_1 = ejecucionPowerGmres(P, alpha, x_0, max_it, tol, alpha-0.1, m)


    eigenvector1, _ = ejecucionPowerEstandar(P, x_0, 5000, tol)

    resta1 = np.array(x_n_1) - np.array(eigenvector1)
    norma1 = np.linalg.norm(resta1)
    print("norma", f"{norma1:.7f}")


    tol = 1e-6

    # Necesitamos un vector inicial x_0
    x_0 = np.random.rand(len(P))
    x_0 = x_0 / np.linalg.norm(np.array(x_0), ord=1)
    x_n_2 = ejecucionPowerGmres(P, alpha, x_0, max_it, tol, alpha-0.1, m)

    eigenvector2, _ = ejecucionPowerEstandar(P, x_0, 20000, tol)

    resta2 = np.array(x_n_2) - np.array(eigenvector2)
    norma2 = np.linalg.norm(resta2)
    print("norma", f"{norma2:.7f}")



    tol = 1e-8

    # Necesitamos un vector inicial x_0
    x_0 = np.random.rand(len(P))
    x_0 = x_0 / np.linalg.norm(np.array(x_0), ord=1)
    x_n_3 = ejecucionPowerGmres(P, alpha, x_0, max_it, tol, alpha-0.1, m)


    eigenvector3, _ = ejecucionPowerEstandar(P, x_0, 5000, tol)

    resta3 = np.array(x_n_3) - np.array(eigenvector3)
    norma3 = np.linalg.norm(resta3)
    print("norma", f"{norma3:.7f}")


