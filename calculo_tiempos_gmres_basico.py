import copy
import threading
import time

import numpy as np

from funciones_comunes import (arreglarNodosColgantes, guardar_diferencias_txt,
                               modificarMatriz, obtenerSolucionPython,
                               residuoDosVectores)
from read_data import read_data, read_data_prueba


def GMRES(A, b, x_0, max_it, tol, vector_solucion_python):
    x = copy.deepcopy(x_0)
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



    N = len(A)
    
    # Creamos el vec r_0, que es b-Ax_0
    r_0 = b - np.dot(A, x_0)
    

    # Generamos una matriz V y una h con todos sus valores a 0
    V = np.zeros((N, max_it+1))
    h = np.zeros((max_it+1, max_it))


    # Establecemos el v_1 al vector inicial normalizado.
    r_0_norm = np.linalg.norm(r_0, ord=2)
    V[:, 0] = r_0 / r_0_norm
        
    # Inicializamos el vector g
    g = np.zeros(max_it+1)
    g[0] = r_0_norm
    
    # Guardamos la norma para no repetir la operación en cada bucle
    b_norm = np.linalg.norm(b, ord=2)
    
    # Vector solucion
    x = np.zeros(N)
    
    # Como trabajamos con matrices a las que accedemos desde el 0, reducimos 1 el número N  y num_cols
    N = N-1
    
    no_convergido = True
    n=0
    while n<=(max_it-1) and no_convergido:

        t = np.dot(A, V[:,n])
        
        # Arnoldi
        for i in range(0, n+1):           
            h[i][n] = np.dot(V[:,i], t)                      
            t = t-np.dot(h[i][n], V[:,i])

        t_norm = np.linalg.norm(t, ord=2)

        h[n+1][n] = t_norm
        V[:,n+1] = t / t_norm
        
        # Givens
        for j in range(0, n):
            c_j = abs(h[j][j]) / (np.sqrt( h[j][j]**2 + h[j+1][j]**2  ))
            s_j = (h[j+1][j] / h[j][j])*c_j

            apply_givens_rotation(h, c_j, s_j, j, n)


        delta = np.sqrt(h[n, n] ** 2 + h[n + 1, n] ** 2)
        c_n = h[n, n] / delta
        s_n = h[n + 1, n] / delta

        h[n][n] = c_n*h[n][n] + s_n*h[n+1][n]
        h[n+1][n] = 0
        
        g[n+1] = -s_n*g[n]
        g[n] = c_n*g[n]
        
        # Comprobamos la convergencia
        conver = abs(g[n+1])/b_norm

        # La matrices con las que hemos estado tratando eran V_{n+1} y H_{n+1}.
        # Para este caso necesitamos V_n y H_n luego las reducimos.
        y = np.linalg.solve(h[:(n+1), :(n+1)], g[:(n+1)])
        x = x_0 + np.dot(V[:, :(n+1)], y)
        x = x / np.linalg.norm(x, ord=1)

        if conver <= tol and n>0:    
            # Para salir del bucle.
            no_convergido = False
        n +=1

    x = x / np.linalg.norm(x, ord=1)
    thread.join(0)
    return x,n


def apply_givens_rotation(h, c, s, k, i):
    temp = c * h[k, i] + s * h[k + 1, i]
    h[k + 1, i] = -s * h[k, i] + c * h[k + 1, i]
    h[k, i] = temp




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

    alpha = 0.95

    M = modificarMatriz(P, alpha)

    N = len(P)
    # Primero formateamos nuestro problema a la forma Ax=b
    # Nuestro sistema es de la forma (I-alphaP)x = (1-alpha)v

    # Nuestro vector b, que en nuestro caso es (1-alpha)v
    v = np.ones(N) / N    
    b = np.dot(1-alpha, v)
    
    # Nuestra matriz, que es A=(I-alpha(P))
    A = np.eye(N) - np.dot(alpha, P)

    # Necesitamos un vector inicial x_0
    # x_0 = np.random.rand(N)
    # x_0 = x_0 / np.linalg.norm(x_0, ord=1)

    x_0 = np.ones(N)/N


    tol = 1e-10
    m=2
    max_it = 10000

    print("--------------- PYTHON --------------")

    vector_propio_python = obtenerSolucionPython(M)
    print(vector_propio_python)


    print("--------------- GMRES --------------")

    start_time1 = time.time()

    x_n, num_it, diferencias = GMRES(A, b, x_0, tol, m, max_it, vector_propio_python)

    end_time1 = time.time()
    elapsed_time1 = end_time1 - start_time1

    print("DIFERENCIAS", diferencias)
    print("TIEMPO", elapsed_time1)
    # print("SOLUCION", x_n)

    guardar_diferencias_txt(diferencias, "gmres.txt")
