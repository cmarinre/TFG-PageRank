import threading
import time

import numpy as np

from funciones_comunes import (arreglarNodosColgantes, guardar_diferencias_txt,
                               modificarMatriz, obtenerSolucionPython,
                               residuoDosVectores)
from gmres_reiniciado import GMRES_m
from read_data import read_data, read_data_prueba


def power_gmres(P, b, alpha, x, max_it, tol, alpha_1, m, vector_solucion_python):


    diferencias = []
    tiempo_inicio = time.time()  # Guardamos el tiempo de inicio de la ejecución del método
    lock = threading.Lock()  # Crear un lock para manejar la sincronización entre hilos
    intervalo_registro = 0.5  # Intervalo de tiempo en segundos entre registros
    ultimo_registro = [time.time()]  # Usamos una lista para permitir modificación dentro del hilo

    def guardar_diferencia():
        while True:
            ahora = time.time()
            if ahora - ultimo_registro[0] >= intervalo_registro:
                diferencia = residuoDosVectores(x, vector_solucion_python)
                with lock:
                    diferencias.append((ahora - tiempo_inicio, diferencia))
                ultimo_registro[0] = ahora

    # Creamos un hilo para guardar la diferencia
    thread = threading.Thread(target=guardar_diferencia)
    thread.daemon = True
    thread.start()


    A = np.eye(len(P)) - np.array(np.dot(alpha, P))
    terminado = False
    # conver = tol+1
    while terminado==False:     
        r=1

        # for i in range(0,2):
        # Aplicación del método GMRES REINICIADO
        x, conver = GMRES_m(A, b, x, m, tol)

        # if conver>tol:
        num_it=0
        while num_it < max_it and r>tol:
            x = x/np.linalg.norm(x, ord=1)
            z = np.dot(P, x)
            r = np.linalg.norm(np.dot(alpha, z) + b - x, ord=2)
            r_0 = r
            r_1 = r
            ratio = 0
            while ratio < alpha_1 and r > tol:
                x = np.dot(alpha, z) + b
                z = np.dot(P, x)
                r = np.linalg.norm(np.dot(alpha, z) + b - x, ord=2)
                ratio = r/r_0
                r_0 = r
            x = np.dot(alpha, z) + b
            x = x/np.linalg.norm(x, ord=1)
            if(r/r_1 > alpha_1):
                num_it = num_it+1

        if(r<tol):
            terminado = True
        # else:
        #     terminado = True

    thread.join(0)
    return x, num_it, diferencias




if __name__ == "__main__":

    P = read_data_prueba("./datos/prueba.mtx")
    # P = read_data("./datos/minnesota2642.mtx")
    # P = read_data("./datos/hollins6012.mtx")
    # P = read_data("./datos/stanford9914.mtx")
    # P = arreglarNodosColgantes(P)

    # P = np.array([[1/2, 1/3, 0, 0],
    #               [0, 1/3, 0, 1],
    #               [0, 1/3, 1/2, 0],
    #               [1/2, 0, 1/2, 0]])

    alpha = 0.99

    M = modificarMatriz(P, alpha)

    N = len(P)


    # Primero formateamos nuestro problema a la forma Ax=b
    # Nuestro sistema es de la forma (I-alphaP)x = (1-alpha)v PERO PASAMOS P y luego lo modificamos

    # Nuestro vector b, que en nuestro caso es (1-alpha)v
    v = np.ones(N) / N    
    b = np.dot(1-alpha, v)


    # Necesitamos un vector inicial x_0
    # x_0 = np.random.rand(N)
    # x_0 = x_0 / np.linalg.norm(np.array(x_0), ord=1)

    x_0 = np.ones(N)/N


    tol = 1e-10
    m = 2
    max_it=100

    print("--------------- PYTHON --------------")

    vector_propio_python = obtenerSolucionPython(M)
    print(vector_propio_python)


    print("--------------- GMRES-POWER m --------------")

    start_time1 = time.time()

    x_n, num_it, diferencias = power_gmres(P, b, alpha, x_0, max_it, tol, alpha-0.1, m, vector_propio_python)

    end_time1 = time.time()
    elapsed_time1 = end_time1 - start_time1

    print("DIFERENCIAS", diferencias)
    print("TIEMPO", elapsed_time1)
    # print("SOLUCION", x_n)

    guardar_diferencias_txt(diferencias, "gmres_power.txt")
