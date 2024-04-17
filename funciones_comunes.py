import time
import numpy as np

from power_method import power_method, power_method_convergence
from power_method_adaptive import adaptive_power_method

def matrizPageRank(n):
    matrix = np.zeros((n, n))

    for i in range(n):
        # Determinar el número de enlaces salientes desde la página i
        num_outlinks = np.random.randint(1, n)  # Puede haber hasta 5 enlaces
        
        # Si no hay enlaces salientes, pasar a la siguiente página
        if num_outlinks == 0:
            random_page = np.random.randint(n)
            matrix[random_page, i] = 1
        else:
            # Asignar probabilidades de enlace iguales a los enlaces salientes
            probabilidad_enlace = 1 / num_outlinks
            outlink_indices = np.random.choice(n, size=num_outlinks, replace=False)
            for j in outlink_indices:
                matrix[j, i] = probabilidad_enlace

    # Normalizar las columnas para que la suma sea igual a 1
    matrix = matrix / np.sum(matrix, axis=0, keepdims=True)
    
    return matrix


def generarMatrizAleatoria(n):
    matrix = np.zeros((n, n))

    # Asignar valores a las conexiones entre páginas
    for i in range(n):
        # Determinar el número de enlaces salientes desde la página i
        num_outlinks = np.random.randint(0, 11)  # Puede haber hasta 10 enlaces
        
        # Si no hay enlaces salientes, pasar a la siguiente página
        if num_outlinks == 0:
            continue
        
        # Asignar la misma probabilidad a todos los enlaces salientes
        probabilidad_enlace = 1 / num_outlinks
        matrix[:, i] = probabilidad_enlace
    
    return matrix


if __name__ == "__main__":
    # Definición de una matriz de ejemplo

    print("Creando matriz")
    A = matrizPageRank(20000)

    # Registro del tiempo de inicio
    start_time1 = time.time()
    # Aplicación del método de las potencias
    eigenvector1, num_it1 = power_method_convergence(A)
    # Registro del tiempo de finalización
    end_time1 = time.time()
    # Cálculo del tiempo transcurrido
    elapsed_time1 = end_time1 - start_time1

    # print("Vector propio correspondiente:", eigenvector)
    print("El tiempo de ejecución de power fue de: {:.5f} segundos".format(elapsed_time1))
    print("Número de iteraciones:", num_it1)
    print("Suma de los valores del vector propio para asegurar que su norma 1 es igual a 1:", np.sum(eigenvector1))
    print("Vector propio:", eigenvector1)

    # Registro del tiempo de inicio
    start_time2 = time.time()

    # Aplicación del método de las potencias adaptado
    eigenvector2, num_it2 = adaptive_power_method(A)
    # Registro del tiempo de finalización
    end_time2 = time.time()
    # Cálculo del tiempo transcurrido
    elapsed_time2 = end_time2 - start_time2

    # print("Vector propio correspondiente:", eigenvector)
    print("El tiempo de ejecución de adaptive fue de: {:.5f} segundos".format(elapsed_time2))
    print("Número de iteraciones:", num_it2)
    print("Suma de los valores del vector propio para asegurar que su norma 1 es igual a 1:", np.sum(eigenvector2))
    print("Vector propio:", eigenvector2)

    # mult1 =np.dot(A,eigenvector1)
    # print("Multiplicacion1:", mult1)

    # mult2 = np.dot(A,eigenvector2)
    # print("Multiplicacion2:", mult2)
