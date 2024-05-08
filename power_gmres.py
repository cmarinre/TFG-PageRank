
import numpy as np
import time
from funciones_comunes import modificarMatriz

from gmres_reiniciado import GMRES_m


def power_gmres(Matriz, M, b, x_0, max_it, tol, alpha_1, m=2):
    no_terminado = True
    conver = tol+1
    while no_terminado:     
        r=1
        it = 0
        while conver>tol and it<m:
            # Aplicación del método GMRES
            x_n, conver = GMRES_m(Matriz, b, x_0, m)
            x_0 = x_n
            it+=1
        x = x_n
        if conver>tol:
            num_it=0
            while num_it < max_it and r>tol:
                x = x/np.linalg.norm(x, ord=1)
                r = np.linalg.norm(np.dot(M, x)-x, ord=2)
                r_0 = r
                r_1 = r
                ratio = 0
                while ratio < alpha_1 and r > tol:
                    x = np.dot(M, x)
                    r = np.linalg.norm(np.dot(M, x)-x, ord=2)
                    ratio = r/r_0
                    r_0 = r
                x = np.dot(M, x)
                x = x/np.linalg.norm(x, ord=1)
                if(r/r_1 > alpha_1):
                    max_it = max_it+1

            if(r<=tol): no_terminado = False
    return x


if __name__ == "__main__":

    # print("Creando matriz")
    # A = matrizPageRank(5)
    # print("matriz creada")

    A = np.array([[1/2, 1/3, 0, 0],
                  [0, 1/3, 0, 1],
                  [0, 1/3, 1/2, 0],
                  [1/2, 0, 1/2, 0]])

    # A = np.array([[0, 1/2, 1/3],
    #         [1/2, 0, 1/3],
    #         [1/2, 1/2, 1/3]])

    alpha = 0.85

    M = modificarMatriz(A, alpha)

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

    tol = 1e-15
    m = 3
    max_it=1000

    start_time = time.time()
    x_n = power_gmres(Matriz, M, b, x_0, max_it, tol, tol, m)

    end_time = time.time()
    elapsed_time = end_time - start_time


    print("El tiempo de ejecución de GMRES fue de: {:.5f} segundos".format(elapsed_time))
    print("Vector solución normalizado", x_n)
