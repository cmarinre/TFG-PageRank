
import numpy as np
import time
from funciones_comunes import matrizPageRank, multiplicacionDosVectores, multiplicacionMatrizVector, multiplicacionValorVector, multiplicacionDosMatrices, multiplicacionValorMatriz
from scipy.sparse.linalg import gmres

def GMRES(A, b, x_0, max_it):

    N = len(A)
    
    # Y nuestro vector r_0, b-Ax_0
    Ax0 = multiplicacionMatrizVector(A, x_0)
    r_0 = np.array(b) - np.array(Ax0)
    
    #Establecemos el máximo de iteraciones
    num_columnas = max_it

    # Generamos una matriz V y una h con todos sus valores a 0
    V = np.zeros((N, num_columnas+1))
    h = np.zeros((num_columnas+1, num_columnas))
    
    # Establecemos el v_1 al vector inicial normalizado.
    r_0_norm = np.linalg.norm(r_0, ord=2)
    V[:, 0] = np.array(r_0 / r_0_norm)
    
    # Inicializamos el vector g
    g = np.zeros(num_columnas+1)
    g[0] = r_0_norm
    
    
    # Vector solucion
    x = np.zeros(N)
    
    # Como trabajamos con matrices a las que accedemos desde el 0, reducimos 1 el número N  y num_cols
    N = N-1
    num_columnas = num_columnas - 1
    
    n=0
    while n<=(num_columnas):

        t = multiplicacionMatrizVector(A, V[:,n])
        
        # Arnoldi
        i=0
        while i <= n:                
            h[i][n] = multiplicacionDosVectores(V[:,i], t)
            aux = multiplicacionValorVector(h[i][n], V[:,i])
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
        
        if n==num_columnas:
            b_norm = np.linalg.norm(b, ord=2)
            h_reducida = h[:(n+1), :(n+1)]
            v_reducida = V[:, :(n+1)]
            g_reducido = g[:(n+1)]
            inversa = np.linalg.inv(h_reducida)
            VnH_1n = multiplicacionDosMatrices(v_reducida, inversa)
            VnH_1ng = multiplicacionMatrizVector(VnH_1n, g_reducido)
            x = x_0 + VnH_1ng
            conver = abs(g[n+1])/b_norm
        n +=1
    
    return x, conver


# La idea de este método es ejecutar n veces el método y poner como
# Vector inicial el vector generado por el anterior GMRES.
def GMRESReiniciado(Matriz, b, x_0, tol, m):
        
    conver = 1
    it=0
    while conver>tol:
        # Aplicación del método GMRES
        x_n, conver = GMRES(Matriz, b, x_0, m)
        x_0 = x_n
        it+=1

    return x_n, m*it





if __name__ == "__main__":

    print("Creando matriz")
    A = matrizPageRank(30)
    print("matriz creada")

    alpha = 0.85
    N = len(A)

    # Primero formateamos nuestro problema a la forma Ax=b
    # Nuestro sistema es de la forma (I-alphaA)x = (1-alpha)v

    # Nuestro vector b, que en nuestro caso es (1-alpha)v
    v = np.ones(N) / N    
    b = multiplicacionValorVector(1-alpha, v)
    
    # Nuestra matriz, que es (I-alpha(A))
    Matriz = np.eye(N) - np.array(multiplicacionValorMatriz(alpha, A))

    # Necesitamos un vector inicial x_0
    x_0 = np.random.rand(N)
    x_0 = x_0 / np.linalg.norm(np.array(x_0), ord=1)

    start_time = time.time()
    x_n, n = GMRESReiniciado(Matriz, b, x_0, 0.000000000001, 100)
    end_time = time.time()
    elapsed_time = end_time - start_time


    print("El tiempo de ejecución de GMRES fue de: {:.5f} segundos".format(elapsed_time))
    
    x_n_norm = x_n / np.linalg.norm(np.array(x_n), ord=1)
    print("Vector solución normalizado", x_n_norm)



    # Para comprobar que funciona, comparamos con un programa ya hecho por pyhton
    x, _ = gmres(Matriz, b, rtol=0.000000000001)
    
    x_norm = x / np.linalg.norm(np.array(x), ord=1)
    print("Vector solución normalizado python", x_norm)
