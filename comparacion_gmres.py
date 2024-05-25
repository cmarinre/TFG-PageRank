import copy
import time

import numpy as np

from funciones_comunes import (arreglarNodosColgantes, modificarMatriz,
                               residuoDosVectores)
from gmres import GMRES
from gmres_reiniciado import GMRESReiniciado
from read_data import read_data


def comparacion_gmres(A, x_0, alpha, max_it, tol, m1, m2, m3):

    N = len(A)

    # Primero formateamos nuestro problema a la forma Ax=b
    # Nuestro sistema es de la forma (I-alphaP)x = (1-alpha)v

    # Nuestro vector b, que en nuestro caso es (1-alpha)v
    v = np.ones(N) / N    
    b = np.dot(1-alpha, v)
    
    # Nuestra matriz, que es (I-alpha(A))
    Matriz = np.eye(N) - np.array(np.dot(alpha, A))
    
    # Necesitamos un vector inicial x_0
    x_0 = np.random.rand(N)
    x_0 = x_0 / np.linalg.norm(np.array(x_0), ord=1)

    # print("comienza GMRES")

    start_time = time.time()
    x_n, iteraciones = GMRES(copy.deepcopy(Matriz), copy.deepcopy(b), copy.deepcopy(x_0), max_it, tol)
    end_time = time.time()
    elapsed_time = end_time - start_time

    # print("El tiempo de ejecución de GMRES fue de: {:.5f} segundos".format(elapsed_time).replace('.', ','))
    print("{:.5f}".format(elapsed_time).replace('.', ','))

    # print("Vector solución", x_n)
    # print("Número de iteraciones:", iteraciones)
    

    # print("comienza GMRES 2")

    start_time_rei = time.time()
    x_n_rei, iteraciones_rei = GMRESReiniciado(copy.deepcopy(Matriz), copy.deepcopy(b), copy.deepcopy(x_0), tol, m1, max_it)
    end_time_rei = time.time()
    elapsed_time_rei = end_time_rei - start_time_rei

    # print("El tiempo de ejecución de GMRES reiniciado 2 fue de: {:.5f} segundos".format(elapsed_time_rei).replace('.', ','))
    print("{:.5f}".format(elapsed_time_rei).replace('.', ','))

    # print("Vector solución", x_n_rei)
    # print("Número de iteraciones:", iteraciones_rei)


    # print("comienza GMRES 5")


    start_time_rei = time.time()
    x_n_rei2, iteraciones_rei = GMRESReiniciado(copy.deepcopy(Matriz), copy.deepcopy(b), copy.deepcopy(x_0), tol, m2, max_it)
    end_time_rei = time.time()
    elapsed_time_rei = end_time_rei - start_time_rei

    # print("El tiempo de ejecución de GMRES reiniciado 5 fue de: {:.5f} segundos".format(elapsed_time_rei).replace('.', ','))
    print("{:.5f}".format(elapsed_time_rei).replace('.', ','))
    # print("Vector solución", x_n_rei2)
    # print("Número de iteraciones:", iteraciones_rei)


    # print("comienza GMRES 8")

    start_time_rei = time.time()
    x_n_rei3, iteraciones_rei = GMRESReiniciado(copy.deepcopy(Matriz), copy.deepcopy(b), copy.deepcopy(x_0), tol, m3, max_it)
    end_time_rei = time.time()
    elapsed_time_rei = end_time_rei - start_time_rei

    # print("El tiempo de ejecución de GMRES reiniciado 8 fue de: {:.5f} segundos".format(elapsed_time_rei).replace('.', ','))
    print("{:.5f}".format(elapsed_time_rei).replace('.', ','))
    # print("Vector solución", x_n_rei3)
    # print("Número de iteraciones:", iteraciones_rei)

    return x_n, x_n_rei, x_n_rei2, x_n_rei3



if __name__ == "__main__":

    P = read_data("./datos/minnesota2642.mtx")
    # P = read_data("./datos/hollins6012.mtx")
    # P = read_data("./datos/stanford9914.mtx")
    P = arreglarNodosColgantes(P)

    # P = np.array([[1/2, 1/3, 0, 0],
    #               [0, 1/3, 0, 1],
    #               [0, 1/3, 1/2, 0],
    #               [1/2, 0, 1/2, 0]])

    alpha = 0.85

    M = modificarMatriz(P, alpha)


    N = len(P)
    x_0 = np.random.rand(N)
    x_0 = x_0 / np.linalg.norm(np.array(x_0), ord=1)

    tol = 1e-4
    m1=2
    m2=5
    m3=8
    max_it = 10000

    # Ejecutamos GMRES, GMRES(2), GMRES(5), GMRES(8)
    x_n, x_n_rei, x_n_rei2, x_n_rei3 = comparacion_gmres(P, x_0, alpha, max_it, tol, m1, m2, m3)
    

    # Calcular los valores y vectores propios por numpy
    valores_propios, vectores_propios = np.linalg.eig(M)
    # Encuentra el índice del valor propio que es igual a 1
    index = np.where(np.isclose(valores_propios, 1))[0][0]
    # Obtiene el vector propio correspondiente
    vector_propio = vectores_propios[:, index]
    vector_propio = vector_propio/np.linalg.norm(vector_propio, ord=1) 



    # Calculamos las normas de las diferencias con el obtenido por el NumPy

    resta1 = residuoDosVectores(x_n, vector_propio)
    norma1 = np.linalg.norm(resta1)

    resta2 = residuoDosVectores(x_n_rei, vector_propio)
    norma2 = np.linalg.norm(resta2)

    resta3 = residuoDosVectores(x_n_rei2, vector_propio)
    norma3 = np.linalg.norm(resta3)

    resta4 = residuoDosVectores(x_n_rei3, vector_propio)
    norma4 = np.linalg.norm(resta4)

    print("{:.8f}".format(norma1).replace('.', ','))
    print("{:.8f}".format(norma2).replace('.', ','))
    print("{:.8f}".format(norma3).replace('.', ','))
    print("{:.8f}".format(norma4).replace('.', ','))

    # print(vector_propio)
    # print(x_n)
    # print(x_n_rei)
    # print(x_n_rei2)
    # print(x_n_rei3)


    print("comienza TOL 6")

    tol = 1e-6
    # Ejecutamos GMRES, GMRES(2), GMRES(5), GMRES(8)
    x_n, x_n_rei, x_n_rei2, x_n_rei3 = comparacion_gmres(P, x_0, alpha, max_it, tol, m1, m2, m3)
    

    # Calculamos las normas de las diferencias con el obtenido por el NumPy

    resta1 = residuoDosVectores(x_n, vector_propio)
    norma1 = np.linalg.norm(resta1)

    resta2 = residuoDosVectores(x_n_rei, vector_propio)
    norma2 = np.linalg.norm(resta2)

    resta3 = residuoDosVectores(x_n_rei2, vector_propio)
    norma3 = np.linalg.norm(resta3)

    resta4 = residuoDosVectores(x_n_rei3, vector_propio)
    norma4 = np.linalg.norm(resta4)


    print("{:.8f}".format(norma1).replace('.', ','))
    print("{:.8f}".format(norma2).replace('.', ','))
    print("{:.8f}".format(norma3).replace('.', ','))
    print("{:.8f}".format(norma4).replace('.', ','))

    # print(vector_propio)
    # print(x_n)
    # print(x_n_rei)
    # print(x_n_rei2)
    # print(x_n_rei3)


    print("comienza TOL 8")

    tol = 1e-8
    # Ejecutamos GMRES, GMRES(2), GMRES(5), GMRES(8)
    x_n, x_n_rei, x_n_rei2, x_n_rei3 = comparacion_gmres(P, x_0, alpha, max_it, tol, m1, m2, m3)
    
    # Calculamos las normas de las diferencias con el obtenido por el NumPy
    resta1 = residuoDosVectores(x_n, vector_propio)
    norma1 = np.linalg.norm(resta1)

    resta2 = residuoDosVectores(x_n_rei, vector_propio)
    norma2 = np.linalg.norm(resta2)

    resta3 = residuoDosVectores(x_n_rei2, vector_propio)
    norma3 = np.linalg.norm(resta3)

    resta4 = residuoDosVectores(x_n_rei3, vector_propio)
    norma4 = np.linalg.norm(resta4)

    print("{:.8f}".format(norma1).replace('.', ','))
    print("{:.8f}".format(norma2).replace('.', ','))
    print("{:.8f}".format(norma3).replace('.', ','))
    print("{:.8f}".format(norma4).replace('.', ','))


    # print(vector_propio)
    # print(x_n)
    # print(x_n_rei)
    # print(x_n_rei2)
    # print(x_n_rei3)

