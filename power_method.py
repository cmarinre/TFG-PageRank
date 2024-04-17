import numpy as np
import time
# from funciones_comunes import generarMatrizAleatoria, matrizPageRank

def power_method(matrix, max_iterations=50000, tolerance=0.00000005):
    # Obtenemos la dimensión de la matriz
    n = len(matrix)
    # Generamos un vector aleatorio de tamaño n
    # vector = np.random.rand(n)
    vector = [0]*n
    vector[1] = 1
    # vector = [1,0,0]
    # Lo normalizamos dividiendo por la norma 1 (suma de las componentes)
    vector /= np.linalg.norm(vector, ord=1)


    # Ejecutamos el método de las potencias y paramos cuando el número de iteraciones sea el máximo
    # O cuando cumple el factor de tolerancia
    for i in range(max_iterations):
        # Multiplicación de la matriz por el vector
        matrix_vector_product = np.dot(matrix, vector)
        # Cálculo del nuevo vector
        new_vector = matrix_vector_product / np.linalg.norm(matrix_vector_product, ord=1)

        # Comprobación de convergencia
        if np.linalg.norm(new_vector - vector, ord=1) < tolerance:
            break

        # Guardamos el vector nuevo
        vector = new_vector
        # print(vector)

    return vector, i

def power_method_convergence(matrix, max_iterations=50000, tolerance=0.000000001):
    # Obtenemos la dimensión de la matriz
    n = len(matrix)
    # Generamos un vector aleatorio de tamaño n
    vector = np.random.rand(n)
    # vector = [0]*n
    # vector[1] = 1    
    # Lo normalizamos dividiendo por la norma 1 (suma de las componentes)
    vector /= np.linalg.norm(vector, ord=1)


    # Ejecutamos el método de las potencias y paramos cuando el número de iteraciones sea el máximo
    # O cuando cumple el factor de tolerancia
    for i in range(max_iterations):
        # Multiplicación de la matriz por el vector
        matrix_vector_product = np.dot(matrix, vector)
        # Cálculo del nuevo vector
        new_vector = matrix_vector_product / np.linalg.norm(matrix_vector_product, ord=1)

        # print("Nuevo vector power method ",new_vector)
        # print("Viejo vector power method ",vector)
        # Comprobación de convergencia
        if all(valor < tolerance for valor in (new_vector-vector)):
            break

        # Guardamos el vector nuevo
        vector = new_vector

    return vector, i


# Ejemplo de uso 
if __name__ == "__main__":
    # Definición de una matriz de ejemplo
    # A = np.array([[1/2, 1/3, 0, 0],
    #               [0, 1/3, 0, 1],
    #               [0, 1/3, 1/2, 0],
    #               [1/2, 0, 1/2, 0]])

    print("Creando matriz")
    # A= matrizPageRank(20000)
    # A= generarMatrizAleatoria(5000)
    print("matriz generada")

    # Registro del tiempo de inicio
    start_time = time.time()

    # Aplicación del método de las potencias
    eigenvector, num_it = power_method(A)

    # Registro del tiempo de finalización
    end_time = time.time()

    # Cálculo del tiempo transcurrido
    elapsed_time = end_time - start_time

    # print("Vector propio correspondiente:", eigenvector)
    print("El tiempo de ejecución fue de: {:.5f} segundos".format(elapsed_time))
    print("Número de iteraciones:", num_it)
    print("Suma de los valores del vector propio para asegurar que su norma 1 es igual a 1:", np.sum(eigenvector))
