
import time

import numpy as np

from funciones_comunes import arreglarNodosColgantes, modificarMatriz
from gmres_reiniciado import GMRES_m


def power_gmres(P, b, alpha, x_0, max_it, tol, alpha_1, m):
    A = np.eye(len(P)) - np.array(np.dot(alpha, P))
    x=x_0
    terminado = False
    # conver = tol+1
    while terminado==False:     
        r=1
        # Aplicación del método GMRES REINICIADO
        x, conver = GMRES_m(A, b, x, m)

        if conver>tol:
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
        else:
            terminado = True

    return x



def power_gmres2(P, b, alpha, x_0, max_it, tol, alpha_1, m):
    A = np.eye(len(P)) - np.array(np.dot(alpha, P))
    x=x_0
    terminado = False
    # conver = tol+1
    while terminado==False:     
        r=1
        for i in range(0,3):
            # Aplicación del método GMRES REINICIADO
            x, conver = GMRES_m(A, b, x, m)

        if conver>tol:
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
        else:
            terminado = True

    return x

# if __name__ == "__main__":


#     P = np.array([[1/2, 1/3, 0, 0],
#                   [0, 1/3, 0, 1],
#                   [0, 1/3, 1/2, 0],
#                   [1/2, 0, 1/2, 0]])

#     alpha = 0.85
#     # P = arreglarNodosColgantes(P)

#     N = len(P)

#     # Primero formateamos nuestro problema a la forma Ax=b
#     # Nuestro sistema es de la forma (I-alphaA)x = (1-alpha)v

#     # Nuestro vector b, que en nuestro caso es (1-alpha)v
#     v = np.ones(N) / N    
#     b = np.dot(1-alpha, v)

#     # Necesitamos un vector inicial x_0
#     x_0 = np.random.rand(N)
#     x_0 = x_0 / np.linalg.norm(np.array(x_0), ord=1)

#     tol = 1e-30
#     m = 2
#     max_it=50

#     start_time = time.time()
#     x_n = power_gmres(P, b, alpha, x_0, max_it, tol, alpha-0.1, m)

#     end_time = time.time()
#     elapsed_time = end_time - start_time


#     print("El tiempo de ejecución de GMRES fue de: {:.5f} segundos".format(elapsed_time))
#     print("Vector solución normalizado", x_n)
