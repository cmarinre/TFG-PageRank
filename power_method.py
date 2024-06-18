import copy
import time

import numpy as np

from funciones_comunes import (arreglarNodosColgantes, modificarMatriz,
                               multiplicacionMatrizVector,
                               obtenerSolucionPython, residuoDosVectores)
from read_data import read_data, read_data_cz1268, read_data_minnesota


# Método de las potencias estándar, calculando la convergencia con la norma 1
# y multiplicando la función del numpy
def power_method(matrix, vector, max_iterations, tolerance):

    # Ejecutamos el método de las potencias y paramos cuando el número de iteraciones sea el máximo
    # O cuando cumple el factor de tolerancia
    for i in range(max_iterations):

        # Multiplicación de la matriz por el vector
        new_vector = np.dot(matrix, vector)
        
        # Aquí deberíamos dividir por la norma pero la norma siempre es 1.
        new_vector = new_vector / np.linalg.norm(new_vector, ord=1)

        # Comprobación de convergencia
        if residuoDosVectores(new_vector, vector) < tolerance:
            break

        # Guardamos el vector nuevo
        vector = new_vector
    return vector, i

# Método de las potencias estándar, calculando la convergencia como la convergencia de las componentes una a una
# y multiplicando la matriz con nuestra función particular.
def power_method_convergence(matrix, vector, max_iterations, tolerance):

    # Ejecutamos el método de las potencias y paramos cuando el número de iteraciones sea el máximo
    # O cuando cumple el factor de tolerancia
    for i in range(max_iterations):
        # Multiplicación de la matriz por el vector
        matrix_vector_product = multiplicacionMatrizVector(matrix, vector)

        #Por si la norma es 0
        if(np.linalg.norm(matrix_vector_product, ord=1)==0): 
            print("No se puede dividir por 0")
            break

        # Cálculo del nuevo vector
        new_vector = matrix_vector_product / np.linalg.norm(matrix_vector_product, ord=1)

       # Comprobación de convergencia
        resta = [abs(new_vector[i] - vector[i]) for i in range(len(vector))]
        if all(valor < tolerance for valor in resta):
            break

        # Guardamos el vector nuevo
        vector = copy.deepcopy(new_vector)

    return vector, i



if __name__ == "__main__":

    P = read_data_minnesota("./datos/minnesota2642.mtx")
    # P = read_data("./datos/hollins6012.mtx")
    # P = read_data("./datos/stanford9914.mtx")
    P = arreglarNodosColgantes(P)


    alpha = 0.85
    M = modificarMatriz(P, alpha)

    N = len(M)
    # x_0 = np.random.rand(N)
    # x_0 = x_0 / np.linalg.norm(np.array(x_0), ord=1)
    x_0 = np.ones(N)/N
    max_it = 10000
    tol=1e-8

    start_time1 = time.time()
    eigenvector1, num_it1 = power_method_convergence(M, x_0, max_it, tol)
    end_time1 = time.time()
    elapsed_time1 = end_time1 - start_time1


    print("El tiempo de ejecución de POWER BÁSICO fue de: {:.5f} segundos".format(elapsed_time1))
    print("Número de iteraciones:", num_it1)
