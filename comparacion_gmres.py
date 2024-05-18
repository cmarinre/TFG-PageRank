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

    print("El tiempo de ejecución de GMRES reiniciado 3 fue de: {:.5f} segundos".format(elapsed_time_rei))
    # print("Vector solución", x_n_rei)
    print("Número de iteraciones:", iteraciones_rei)


    print("comienza GMRES 5")


    start_time_rei = time.time()
    x_n_rei2, iteraciones_rei = GMRESReiniciado(copy.deepcopy(Matriz), copy.deepcopy(b), copy.deepcopy(x_0), tol, m2, max_it)
    end_time_rei = time.time()
    elapsed_time_rei = end_time_rei - start_time_rei

    print("El tiempo de ejecución de GMRES reiniciado 5 fue de: {:.5f} segundos".format(elapsed_time_rei))
    # print("Vector solución", x_n_rei)
    print("Número de iteraciones:", iteraciones_rei)


    print("comienza GMRES 8")

    start_time_rei = time.time()
    x_n_rei3, iteraciones_rei = GMRESReiniciado(copy.deepcopy(Matriz), copy.deepcopy(b), copy.deepcopy(x_0), tol, m3, max_it)
    end_time_rei = time.time()
    elapsed_time_rei = end_time_rei - start_time_rei

    print("El tiempo de ejecución de GMRES reiniciado 8 fue de: {:.5f} segundos".format(elapsed_time_rei))
    # print("Vector solución", x_n_rei)
    print("Número de iteraciones:", iteraciones_rei)

    return x_n, x_n_rei, x_n_rei2, x_n_rei3



if __name__ == "__main__":

    # P = read_data("./datos/minnesota2642.mtx")
    # P = read_data("./datos/hollins6012.mtx")
    P = read_data("./datos/stanford9914.mtx")
    A = arreglarNodosColgantes(P)
    
    N = len(A)
    x_0 = np.random.rand(N)
    x_0 = x_0 / np.linalg.norm(np.array(x_0), ord=1)

    x_n, x_n_rei, x_n_rei2, x_n_rei3 = comparacion_gmres(A, x_0, 0.85, 500, 1e-8, 2, 5, 8)
    
    # eigenvector1, num_it1 = ejecucionPowerEstandar(A, x_0, 500, 0.00000001)


    # Calcular los valores y vectores propios
    valores_propios, vectores_propios = np.linalg.eig(A)

    # Encontrar el índice del valor propio más cercano a 1
    indice = np.argmin(np.abs(valores_propios - 1))

    # Obtener el vector propio asociado al valor propio 1
    vector_propio = vectores_propios[:, indice]
    vector_propio = vector_propio/np.linalg.norm(vector_propio, ord=1) 


    resta = np.array(x_n) - np.array(vector_propio)
    norma = np.linalg.norm(resta)

    resta2 = np.array(x_n_rei) - np.array(vector_propio)
    norma2 = np.linalg.norm(resta2)

    resta3 = np.array(x_n_rei2) - np.array(vector_propio)
    norma3 = np.linalg.norm(resta3)

    resta4 = np.array(x_n_rei3) - np.array(vector_propio)
    norma4 = np.linalg.norm(resta4)

    print(norma)
    print(norma2)
    print(norma3)
    print(norma4)
