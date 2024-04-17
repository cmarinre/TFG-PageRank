import numpy as np
import time
# from funciones_comunes import generarMatrizAleatoria, matrizPageRank

def adaptive_power_method(matrix, max_iterations=50000, tolerance=0.000000001):

    # Obtenemos la dimensión de la matriz
    n = len(matrix)

    # Vector de convergencia de componentes
    converg_comp = [0] * n
    converg_comp_final = [1] * n

    # Generamos un vector aleatorio de tamaño n
    vector = np.random.rand(n)
    # vector = [0]*n
    # vector[1] = 1
    # Lo normalizamos dividiendo por la norma 1 (suma de las componentes)
    vector /= np.linalg.norm(vector, ord=1)

    # Inicializamos los dos vectores y la matriz
    x_k = vector
    x_kii = [0] * n
    matrix_Aii = matrix


    # En cada iteración
    for j in range(max_iterations):

        # print("antes", x_k)
        # Calculamos el nuevo vector
        x_k1 = np.dot(matrix_Aii, x_k) + x_kii


        # Dividimos por su norma, que será 1.
        # print(np.linalg.norm(x_k1, ord=1))
        # x_k1 = x_k1 / np.linalg.norm(x_k1, ord=1)

        # Comprobamos componente por componente si ha cumplido el criterio de convergencia.
        # En los que lo haya cumplido, la fila la ponemos a 0
        # Y en el nuevo vector ponemos el valor y ya no lo tocamos nunca más
        
        # Para cada componente
        for i in range(len(vector)):
            # Si la componente no había cumplido ya el criterio de convergencia
            if converg_comp[i]==0:
                # Si la cumple ahora
                resta = abs(x_k1[i] - x_k[i])
                if  resta < tolerance:
                    # Lo apuntamos en el vector de convergencia
                    converg_comp[i] = 1
                    # Ponemos en el x_kii a su valor
                    x_kii[i] = x_k1[i]
                    # Y en la matriz esa fila a 0
                    matrix_Aii[i] = [0] * len(matrix_Aii[i])

        # print("Nuevo vector adaptive method ",x_k1)
        # print("El que ya ha cumplido vector adaptive method ",x_kii)
        # print("Viejo vector adaptive method ",x_k)
        
        # Guardamos el vector nuevo
        x_k = x_k1

        # Comprobación de convergencia
        if converg_comp==converg_comp_final :
            break


    return x_k, j


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
    eigenvector, num_it = adaptive_power_method(A)

    # Registro del tiempo de finalización
    end_time = time.time()

    # Cálculo del tiempo transcurrido
    elapsed_time = end_time - start_time

    # print("Vector propio correspondiente:", eigenvector)
    print("El tiempo de ejecución fue de: {:.5f} segundos".format(elapsed_time))
    print("Número de iteraciones:", num_it)
    print("Suma de los valores del vector propio para asegurar que su norma 1 es igual a 1:", np.sum(eigenvector))
