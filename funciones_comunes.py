import time
import numpy as np

from power_method import power_method, power_method_convergence
from power_method_adaptive import adaptive_power_method

# Función para la generación de una matriz dado un tamaño, que se parezca a una matriz de enlaces.
# Intentamos establecer el número de enlaces salientes de cada página y ponerlos aleatoriamente en páginas distintas.
def matrizPageRank(n):
    matrix = np.zeros((n, n))

    for i in range(n):
        # Determinar el número de enlaces salientes desde la página i
        num_outlinks = np.random.randint(1, n)  # Puede haber hasta n enlaces
        
        # Si no hay enlaces salientes, le ponemos en una aleatoria un 1
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


# Función que compara los métodos de potencias y potencias adaptado, mirando la convergencia del primero
# como la convergencia de la norma.
def comparacionPowers(A):

    # Registro del tiempo de inicio
    start_time1 = time.time()
    # Aplicación del método de las potencias
    eigenvector1, num_it1 = power_method(A)
    # Registro del tiempo de finalización
    end_time1 = time.time()
    # Cálculo del tiempo transcurrido
    elapsed_time1 = end_time1 - start_time1

    print("El tiempo de ejecución de power fue de: {:.5f} segundos".format(elapsed_time1))
    print("Número de iteraciones:", num_it1)
    print("Suma de los valores del vector propio para asegurar que su norma 1 es igual a 1:", np.sum(eigenvector1))
    print("Vector propio:", eigenvector1[0])

    # Registro del tiempo de inicio
    start_time2 = time.time()

    # Aplicación del método de las potencias adaptado
    eigenvector2, num_it2 = adaptive_power_method(A)
    # Registro del tiempo de finalización
    end_time2 = time.time()
    # Cálculo del tiempo transcurrido
    elapsed_time2 = end_time2 - start_time2

    print("El tiempo de ejecución de adaptive fue de: {:.5f} segundos".format(elapsed_time2))
    print("Número de iteraciones:", num_it2)
    print("Suma de los valores del vector propio para asegurar que su norma 1 es igual a 1:", np.sum(eigenvector2))
    print("Vector propio:", eigenvector2[0])



# Función que compara los métodos de potencias y potencias adaptado, mirando la convergencia del primero
# como la convergencia de sus componentes.
def comparacionPowersMult(A):

    # Registro del tiempo de inicio
    start_time1 = time.time()
    # Aplicación del método de las potencias
    eigenvector1, num_it1 = power_method_convergence(A)
    # Registro del tiempo de finalización
    end_time1 = time.time()
    # Cálculo del tiempo transcurrido
    elapsed_time1 = end_time1 - start_time1

    print("El tiempo de ejecución de power fue de: {:.5f} segundos".format(elapsed_time1))
    print("Número de iteraciones:", num_it1)
    print("Suma de los valores del vector propio para asegurar que su norma 1 es igual a 1:", np.sum(eigenvector1))
    print("Vector propio:", eigenvector1[1])

    # Registro del tiempo de inicio
    start_time2 = time.time()
    # Aplicación del método de las potencias adaptado
    eigenvector2, num_it2 = adaptive_power_method(A)
    # Registro del tiempo de finalización
    end_time2 = time.time()
    # Cálculo del tiempo transcurrido
    elapsed_time2 = end_time2 - start_time2

    print("El tiempo de ejecución de adaptive fue de: {:.5f} segundos".format(elapsed_time2))
    print("Número de iteraciones:", num_it2)
    print("Suma de los valores del vector propio para asegurar que su norma 1 es igual a 1:", np.sum(eigenvector2))
    print("Vector propio:", eigenvector2[1])



if __name__ == "__main__":
    # A = np.array([[1/2, 1/3, 0, 0],
    #               [0, 1/3, 0, 1],
    #               [0, 1/3, 1/2, 0],
    #               [1/2, 0, 1/2, 0]])

    print("Creando matriz")
    A = matrizPageRank(1500)
    print("Matriz creada")

    comparacionPowersMult(A)
    comparacionPowers(A)
    

