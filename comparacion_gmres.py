import copy
import time

import numpy as np

from comparacion_powers import ejecucionPowerEstandar
from funciones_comunes import (arreglarNodosColgantes, matrizPageRank,
                               modificarMatriz)
from gmres import GMRES
from gmres_reiniciado import GMRESReiniciado
from power_method import power_method
from read_data import read_data


def comparacion_gmres(A, x_0, alpha, max_it, tol, m1, m2, m3):

    N = len(A)

    # Primero formateamos nuestro problema a la forma Ax=b
    # Nuestro sistema es de la forma (I-alphaA)x = (1-alpha)v

    # Nuestro vector b, que en nuestro caso es (1-alpha)v
    v = np.ones(N) / N    
    b = np.dot(1-alpha, v)
    
    # Nuestra matriz, que es (I-alpha(A))
    Matriz = np.eye(N) - np.array(np.dot(alpha, A))
    
    # Necesitamos un vector inicial x_0
    x_0 = np.random.rand(N)
    x_0 = x_0 / np.linalg.norm(np.array(x_0), ord=1)

    print("comienza GMRES")

    start_time = time.time()
    x_n, iteraciones = GMRES(copy.deepcopy(Matriz), copy.deepcopy(b), copy.deepcopy(x_0), max_it, tol)
    end_time = time.time()
    elapsed_time = end_time - start_time

    print("El tiempo de ejecución de GMRES fue de: {:.5f} segundos".format(elapsed_time))
    # print("Vector solución", x_n)
    print("Número de iteraciones:", iteraciones)
    

    print("comienza GMRES 2")

    start_time_rei = time.time()
    x_n_rei, iteraciones_rei = GMRESReiniciado(copy.deepcopy(Matriz), copy.deepcopy(b), copy.deepcopy(x_0), tol, m1, max_it)
    end_time_rei = time.time()
    elapsed_time_rei = end_time_rei - start_time_rei

    print("El tiempo de ejecución de GMRES reiniciado fue de: {:.5f} segundos".format(elapsed_time_rei))
    # print("Vector solución", x_n_rei)
    print("Número de iteraciones:", iteraciones_rei)


    print("comienza GMRES 5")


    start_time_rei = time.time()
    x_n_rei2, iteraciones_rei = GMRESReiniciado(copy.deepcopy(Matriz), copy.deepcopy(b), copy.deepcopy(x_0), tol, m2, max_it)
    end_time_rei = time.time()
    elapsed_time_rei = end_time_rei - start_time_rei

    print("El tiempo de ejecución de GMRES reiniciado fue de: {:.5f} segundos".format(elapsed_time_rei))
    # print("Vector solución", x_n_rei)
    print("Número de iteraciones:", iteraciones_rei)


    # print("comienza GMRES 8")

    # start_time_rei = time.time()
    # x_n_rei3, iteraciones_rei = GMRESReiniciado(copy.deepcopy(Matriz), copy.deepcopy(b), copy.deepcopy(x_0), tol, m3, max_it)
    # end_time_rei = time.time()
    # elapsed_time_rei = end_time_rei - start_time_rei

    # print("El tiempo de ejecución de GMRES reiniciado fue de: {:.5f} segundos".format(elapsed_time_rei))
    # # print("Vector solución", x_n_rei)
    # print("Número de iteraciones:", iteraciones_rei)

    return x_n, x_n_rei, x_n_rei2, x_n_rei3



if __name__ == "__main__":

    A = read_data("./datos/stanford9914.mtx")
    A = arreglarNodosColgantes(A)


    # A = np.array([[1/2, 1/3, 0, 0],
    #         [0, 1/3, 0, 1],
    #         [0, 1/3, 1/2, 0],
    #         [1/2, 0, 1/2, 0]])
    
    N = len(A)
    x_0 = np.random.rand(N)
    x_0 = x_0 / np.linalg.norm(np.array(x_0), ord=1)

    x_n, x_n_rei, x_n_rei2, x_n_rei3 = comparacion_gmres(A, x_0, 0.85, 500, 0.00000001, 2, 5, 8)
    
    # eigenvector1, num_it1 = ejecucionPowerEstandar(A, x_0, 500, 0.0001)

    # resta = np.array(x_n) - np.array(eigenvector1)
    # norma = np.linalg.norm(resta)

    # resta2 = np.array(x_n_rei) - np.array(eigenvector1)
    # norma2 = np.linalg.norm(resta2)

    # resta3 = np.array(x_n_rei2) - np.array(eigenvector1)
    # norma3 = np.linalg.norm(resta3)

    # resta4 = np.array(x_n_rei3) - np.array(eigenvector1)
    # norma4 = np.linalg.norm(resta4)

    # print(norma)
    # print(norma2)
    # print(norma3)
    # print(norma4)
