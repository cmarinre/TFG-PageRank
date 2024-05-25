import threading
import time

import numpy as np

from funciones_comunes import (arreglarNodosColgantes, guardar_diferencias_txt,
                               modificarMatriz, obtenerSolucionPython,
                               residuoDosVectores)
from read_data import read_data, read_data_prueba

# Método de las potencias estándar al que le añadimos guardar cada x tiempo, la diferencia con el vector "óptimo" que buscamos

def power_method(matrix, vector, max_iterations, tolerance, vector_solucion_python):

    diferencias = []
    tiempo_inicio = time.time()  # Guardamos el tiempo de inicio de la ejecución del método
    lock = threading.Lock()  # Crear un lock para manejar la sincronización entre hilos
    intervalo_registro = 0.5  # Intervalo de tiempo en segundos entre registros
    ultimo_registro = [time.time()]  # Usamos una lista para permitir modificación dentro del hilo

    def guardar_diferencia():
        while True:
            ahora = time.time()
            if ahora - ultimo_registro[0] >= intervalo_registro:
                diferencia = residuoDosVectores(vector, vector_solucion_python)
                with lock:
                    diferencias.append((ahora - tiempo_inicio, diferencia))
                ultimo_registro[0] = ahora

    # Creamos un hilo para guardar la diferencia
    thread = threading.Thread(target=guardar_diferencia)
    thread.daemon = True
    thread.start()


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

    thread.join(0)
    return vector, i, diferencias






if __name__ == "__main__":

    # P = read_data_prueba("./datos/prueba.mtx")
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
    # x_0 = np.random.rand(N)
    # x_0 = x_0 / np.linalg.norm(np.array(x_0), ord=1)

    x_0 = np.ones(N)/N


    tol = 1e-10
    max_it = 10000

    print("--------------- PYTHON --------------")

    vector_propio_python = obtenerSolucionPython(M)
    print(vector_propio_python)



    print("--------------- POWER --------------")

    start_time1 = time.time()

    x_n, num_it, diferencias = power_method(M, x_0, max_it, tol, vector_propio_python)

    end_time1 = time.time()
    elapsed_time1 = end_time1 - start_time1

    print("DIFERENCIAS", diferencias)
    print("TIEMPO", elapsed_time1)
    # print("SOLUCION", x_n)

    guardar_diferencias_txt(diferencias, "power.txt")
