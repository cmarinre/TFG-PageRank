# import copy
# import threading
# import time

# import numpy as np

# from comparacionlineal_power_gmres import GMRESReiniciado, power_method
# from funciones_comunes import (arreglarNodosColgantes, modificarMatriz,
#                                obtenerSolucionPython, residuoDosVectores)
# from gmres_reiniciado import GMRES_m
# from read_data import read_data


# def power_gmres(P, b, alpha, x, max_it, tol, alpha_1, m, vector_solucion_python):

#     x_sol = x
#     diferencias = []
#     tiempo_inicio = time.time()  # Guardamos el tiempo de inicio de la ejecución del método
#     lock = threading.Lock()  # Crear un lock para manejar la sincronización entre hilos
#     intervalo_registro = 0.1  # Intervalo de tiempo en segundos entre registros
#     ultimo_registro = [time.time()]  # Usamos una lista para permitir modificación dentro del hilo

#     def guardar_diferencia():
#         while True:
#             ahora = time.time()
#             if ahora - ultimo_registro[0] >= intervalo_registro:
#                 diferencia = residuoDosVectores(x_sol, vector_solucion_python)
#                 with lock:
#                     diferencias.append((ahora - tiempo_inicio, diferencia))
#                 ultimo_registro[0] = ahora

#     # Creamos un hilo para guardar la diferencia
#     thread = threading.Thread(target=guardar_diferencia)
#     thread.daemon = True
#     thread.start()


#     A = np.eye(len(P)) - np.array(np.dot(alpha, P))
#     terminado = False
#     # conver = tol+1
#     while terminado==False:     
#         r=1
#         for i in range(0,2):
#             # Aplicación del método GMRES REINICIADO
#             x_sol, conver = GMRES_m(A, b, x_sol, m)

#         x = copy.deepcopy(x_sol)
#         if conver>tol:
#             num_it=0
#             while num_it < max_it and r>tol:
#                 x = x/np.linalg.norm(x, ord=1)
#                 z = np.dot(P, x)
#                 r = np.linalg.norm(np.dot(alpha, z) + b - x, ord=2)
#                 r_0 = r
#                 r_1 = r
#                 ratio = 0
#                 while ratio < alpha_1 and r > tol:
#                     x = np.dot(alpha, z) + b
#                     z = np.dot(P, x)
#                     r = np.linalg.norm(np.dot(alpha, z) + b - x, ord=2)
#                     ratio = r/r_0
#                     r_0 = r
#                 x = np.dot(alpha, z) + b
#                 x_sol = x/np.linalg.norm(x, ord=1)
#                 if(r/r_1 > alpha_1):
#                     num_it = num_it+1

#             if(r<tol):
#                 terminado = True
#         else:
#             terminado = True

#     thread.join(0)
#     return x, diferencias


# def guardar_diferencias_txt(diferencias, filename):
#     with open(filename, 'w') as f:
#         for tiempo, diferencia in diferencias:
#             tiempo_str = str(tiempo).replace('.', ',')
#             diferencia_str = str(diferencia).replace('.', ',')
#             f.write(f"{tiempo_str},{diferencia_str}\n")


# if __name__ == "__main__":

#     P = read_data("./datos/minnesota2642.mtx")
#     # P = read_data("./datos/hollins6012.mtx")
#     # P = read_data("./datos/stanford9914.mtx")
#     P = arreglarNodosColgantes(P)



#     # P = np.array([[1/2, 1/3, 0, 0],
#     #               [0, 1/3, 0, 1],
#     #               [0, 1/3, 1/2, 0],
#     #               [1/2, 0, 1/2, 0]])

#     alpha = 0.99998

#     M = modificarMatriz(P, alpha)

#     N = len(P)
#     x_0 = np.random.rand(N)
#     x_0 = x_0 / np.linalg.norm(np.array(x_0), ord=1)

#     tol = 1e-8
#     m=2
#     max_it = 20

#     print("--------------- PYTHON --------------")

#     vector_propio_python = obtenerSolucionPython(M)
#     print(vector_propio_python)




#     print("--------------- POWER --------------")

#     start_time1 = time.time()

#     x_n, num_it, diferencias = power_method(M, x_0, max_it*100, tol, vector_propio_python)

#     end_time1 = time.time()
#     elapsed_time1 = end_time1 - start_time1

#     # print("DIFERENCIAS", diferencias)
#     print("TIEMPO", elapsed_time1)
#     # print("SOLUCION", x_n)

#     guardar_diferencias_txt(diferencias, "power.txt")


#     print("--------------- GMRES(2) --------------")

#     # Nuestro vector b, que en nuestro caso es (1-alpha)v
#     v = np.ones(N) / N    
#     b = np.dot(1-alpha, v)
    
#     # Nuestra matriz, que es A=(I-alpha(P))
#     A = np.eye(N) - np.array(np.dot(alpha, P))

#     start_time1 = time.time()

#     x_n, num_it, diferencias2 = GMRESReiniciado(A, b, x_0, tol, m, max_it*100, vector_propio_python)

#     end_time1 = time.time()
#     elapsed_time1 = end_time1 - start_time1


#     # print("DIFERENCIAS", diferencias2)
#     print("TIEMPO", elapsed_time1)
#     # print("SOLUCION", x_n)
#     guardar_diferencias_txt(diferencias2, "gmres.txt")






#     print("--------------- POWER-GMRES(2) --------------")


#     start_time1 = time.time()

#     x_n, diferencias3 = power_gmres(P, b, alpha, x_0, max_it, tol, alpha-1, m, vector_propio_python)
    
#     end_time1 = time.time()
#     elapsed_time1 = end_time1 - start_time1


#     # print("DIFERENCIAS", diferencias3)
#     print("TIEMPO", elapsed_time1)
#     # print("SOLUCION", x_n)
#     guardar_diferencias_txt(diferencias3, "gmres_power.txt")


