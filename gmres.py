
import time

import numpy as np


def GMRES(A, b, x_0, max_it, tol):

    N = len(A)
    
    # Creamos el vec r_0, que es b-Ax_0
    Ax0 = np.dot(A, x_0)
    r_0 = np.array(b) - np.array(Ax0)
    
    # Establecemos el número de columnas inicial a 2
    num_columnas = 2

    # Generamos una matriz V y una h con todos sus valores a 0
    V = np.zeros((N, num_columnas))
    h = np.zeros((num_columnas, num_columnas-1))


    # Establecemos el v_1 al vector inicial normalizado.
    r_0_norm = np.linalg.norm(r_0, ord=2)
    V[:, 0] = np.array(r_0 / r_0_norm)
        
    # Inicializamos el vector g
    g = np.zeros(num_columnas)
    g[0] = r_0_norm
    
    # Guardamos la norma para no repetir la operación en cada bucle
    b_norm = np.linalg.norm(b, ord=2)
    
    # Vector solucion
    x = np.zeros(N)
    
    # Como trabajamos con matrices a las que accedemos desde el 0, reducimos 1 el número N  y num_cols
    N = N-1
    num_columnas = num_columnas - 1
    
    no_convergido = True
    n=0
    while n<=(max_it-1) and no_convergido:

        t = np.dot(A, V[:,n])
        
        # Arnoldi
        i=0
        while i <= n:                
            h[i][n] = np.dot(V[:,i], t)
            aux = np.dot(h[i][n], V[:,i])
            t = [t[k] - aux[k] for k in range(min(len(t), len(aux)))]
            i+=1
        t_norm = np.linalg.norm(t, ord=2)
        h[n+1][n] = t_norm
        V[:,n+1] = t / t_norm
        
        # Givens
        j=0
        while j<=n-1:     
            c_j = abs(h[j][j]) / (np.sqrt( h[j][j]*h[j][j] + h[j+1][j]*h[j+1][j]  ))
            s_j = (h[j+1][j] / h[j][j])*c_j
            aux1 = c_j*h[j][n] + s_j*h[j+1][n]
            aux2 = -s_j*h[j][n] + c_j*h[j+1][n]
            h[j][n] = aux1
            h[j+1][n] = aux2
            j += 1

        c_n = abs(h[n][n]) / (np.sqrt( h[n][n]*h[n][n] + h[n+1][n]*h[n+1][n]  ))
        s_n = (h[n+1][n] / h[n][n])*c_n
        h[n][n] = c_n*h[n][n] + s_n*h[n+1][n]
        h[n+1][n] = 0
        
        g[n+1] = -s_n*g[n]
        g[n] = c_n*g[n]
        
        # Comprobamos la convergencia
        conver = abs(g[n+1])/b_norm

        if conver <= tol and n>0:
            # La matrices con las que hemos estado tratando eran V_{n+1} y H_{n+1}.
            # Para este caso necesitamos V_n y H_n luego las reducimos.
            h_reducida = h[:(n+1), :(n+1)]
            v_reducida = V[:, :(n+1)]
            # Lo mismo con el vector g
            g_reducido = g[:(n+1)]
            # Calculamos x = x_0 + V_{n} H^{-1}_{n} g 
            inversa = np.linalg.inv(h_reducida)
            VnH_1n = np.dot(v_reducida, inversa)
            VnH_1ng = np.dot(VnH_1n, g_reducido)
            x = x_0 + VnH_1ng
            # Para salir del bucle.
            no_convergido = False


        # Aumentar el tamaño de la matriz V
        # Creamos columna a 0
        nueva_col = np.zeros((N+1, 1))
        # La añadimos
        V = np.hstack((V, nueva_col))

        # Aumentar el tamaño de la matriz h
        # Añadir una nueva fila al final
        nueva_fila = np.zeros((1, num_columnas))
        h = np.vstack((h, nueva_fila))

        # Añadir una nueva columna al final
        nueva_columna = np.zeros((num_columnas+2, 1))
        h = np.hstack((h, nueva_columna))

        # Aumentar el tamaño del vector g
        g = np.append(g, [0])

        #Aumentamos el número de columnas y la iteración
        num_columnas += 1
        n +=1

    
    x = x / np.linalg.norm(x, ord=1)
    return x,n




if __name__ == "__main__":

    # print("Creando matriz")
    # A = matrizPageRank(3)
    # print("matriz creada")

    A = np.array([[1/2, 1/3, 0, 0],
            [0, 1/3, 0, 1],
            [0, 1/3, 1/2, 0],
            [1/2, 0, 1/2, 0]])

    alpha = 0.85
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

    # Registro del tiempo de inicio
    start_time = time.time()
    # Aplicación del método GMRES
    x_n, it = GMRES(Matriz, b, x_0, 100, 0.000000000001)
    # Registro del tiempo de finalización
    end_time = time.time()
    # Cálculo del tiempo transcurrido
    elapsed_time = end_time - start_time


    print("El tiempo de ejecución de GMRES fue de: {:.5f} segundos".format(elapsed_time))
    print("Vector solución", x_n)

