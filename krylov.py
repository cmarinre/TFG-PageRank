
import numpy as np
from funciones_comunes import matrizPageRank, sumaDosMatrices, multiplicacionDosVectores, multiplicacionMatrizVector, multiplicacionValorVector, multiplicacionDosMatrices, modificarMatriz, multiplicacionValorMatriz
from scipy.sparse.linalg import gmres

# Dada una matriz A y un vector v, generará una base ortonormal del 
# subespacio de Krylov K_n(A,v). Generará n vectores, ortonormales entre sí. 
# Estarán en la matriz V. En la matriz h quedan los escalares que nos ayudan a encontrar estos vectores.
def arnoldi(A, v, num_columnas):
    N = len(A)
    # Generamos una matriz V y una h con todos sus valores a 0
    V = [[0] * (N) for _ in range(num_columnas)]
    h = [[0] * (num_columnas-1) for _ in range(num_columnas)]
    # Establecemos el v_1 al vector inicial normalizado.
    V[0] = v / np.linalg.norm(v, ord=2)
    
    # print(N-1)
    n=0
    while n<(num_columnas-1):
        t = multiplicacionMatrizVector(A, V[n])
        i=0
        while i <= n:    
            h[i][n] = multiplicacionDosVectores(V[i], t)
            aux = multiplicacionValorVector(h[i][n], V[i])
            t = [t[j] - aux[j] for j in range(min(len(t), len(aux)))]
            i+=1
            
        Vs_norm = np.linalg.norm(t, ord=2)  
        # print("n+1", n+1) 
        h[n+1][n] = Vs_norm
        # print("H", h) 
        V[n+1] = t / Vs_norm
        # print("V", V)

        n +=1
    # print(n)
    # print(i)
    return V,h

def rotacionesGivens(H, r0):

    N = len(H)
    h = H

    print("h", h)
    g = [0]*(N)
    g[0] = r0
    for n in range(0,N):  
        for i in range(0,n):
            c_i = abs(h[i][i]) / (np.sqrt( h[i][i]*h[i][i] + h[i+1][i]*h[i+1][i]  ))
            s_i = (h[i+1][i] / h[i][i])*c_i
            h[i][n] = c_i*h[i][n] + s_i*h[i+1][n]
            h[i+1][n] = -s_i*h[i][n] + c_i*h[i+1][n]
        c_n = abs(h[n][n]) / (np.sqrt( h[n][n]*h[n][n] + h[n+1][n]*h[n+1][n]  ))
        s_n = (h[n+1][n] / h[n][n])*c_n
        h[n][n] = c_n*h[n][n] + s_n*h[n+1][n]
        h[n+1][n] = 0
        g[n] = c_n*g[n]
        g[n+1] = -s_n*g[n]
    h = h[:-1, :]


def GMRES(A, alpha, max_it, tol):


    M = modificarMatriz(A, alpha)
    N = len(A)

    # Ax=b
    
    # Nuestro sistema es de la forma (I-alphaA)x = (1-alpha)v

    # Necesitamos un vector inicial x_0
    x_0 = np.random.rand(N)

    # Nuestro vector b, que en nuestro caso es (1-alpha)v
    v = np.ones(N) / N
    b = multiplicacionValorVector(1-alpha, v)
    print("b", b)
    
    # Nuestra matriz, que es (I-alpha(A))
    Matriz = np.eye(N) - np.array(multiplicacionValorMatriz(alpha, A))
    print("matriz", Matriz)

    # Y nuestro vector r_0, b-Ax_0
    Matx0 = multiplicacionMatrizVector(Matriz, x_0)
    print("Matx0", Matx0)

    r_0 = np.array(b) - np.array(Matx0)
    print("r_0", r_0)

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
 
    # Guardamos la norma para no repetir la operación en cada bucle
    b_norm = np.linalg.norm(b, ord=2)

    # Vector solucion
    x = np.zeros(N)

    N = N-1
    num_columnas = num_columnas - 1


    n=0
    while n<=(num_columnas):
        print("n", n)
        t = multiplicacionMatrizVector(Matriz, V[:,n])
        i=0
        while i <= n:    
            # print("i", i)
            h[i][n] = multiplicacionDosVectores(V[:,i], t)
            aux = multiplicacionValorVector(h[i][n], V[:,i])
            t = [t[k] - aux[k] for k in range(min(len(t), len(aux)))]
            i+=1
        t_norm = np.linalg.norm(t, ord=2)
        h[n+1][n] = t_norm
        V[:,n+1] = t / t_norm
        n +=1

    print(V)

    n=0
    while n<=(num_columnas):
        j=0
        while j<=n-1:
            # print("j", j)
            c_j = abs(h[j][j]) / (np.sqrt( h[j][j]*h[j][j] + h[j+1][j]*h[j+1][j]  ))
            s_j = (h[j+1][j] / h[j][j])*c_j
            h[j][n] = c_j*h[j][n] + s_j*h[j+1][n]
            h[j+1][n] = -s_j*h[j][n] + c_j*h[j+1][n]
            j += 1
            
        c_n = abs(h[n][n]) / (np.sqrt( h[n][n]*h[n][n] + h[n+1][n]*h[n+1][n]  ))
        s_n = (h[n+1][n] / h[n][n])*c_n
        h[n][n] = c_n*h[n][n] + s_n*h[n+1][n]
        h[n+1][n] = 0

        g[n] = c_n*g[n]
        g[n+1] = -s_n*g[n]

        conver = abs(g[n+1])
        print("conver", conver)
        if conver <= tol and n>0:
            h_reducida = h[:n, :n]
            v_reducida = V[:, :n]
            print(v_reducida)
            g_reducido = g[:n]
            inversa = np.linalg.inv(h_reducida)
            VnH_1n = multiplicacionDosMatrices(v_reducida, inversa)
            x = r_0 + multiplicacionMatrizVector(VnH_1n, g_reducido)
            print("x", x)
        n +=1
    
    return x







def GMRES2(A, alpha, max_it, tol):


    M = modificarMatriz(A, alpha)
    N = len(A)

    # Ax=b
    
    # Nuestro sistema es de la forma (I-alphaA)x = (1-alpha)v

    # Necesitamos un vector inicial x_0
    x_0 = np.random.rand(N)

    # Nuestro vector b, que en nuestro caso es (1-alpha)v
    v = np.ones(N) / N
    b = multiplicacionValorVector(1-alpha, v)
    print("b", b)
    
    # Nuestra matriz, que es (I-alpha(A))
    Matriz = np.eye(N) - np.array(multiplicacionValorMatriz(alpha, A))
    print("matriz", Matriz)

    # Y nuestro vector r_0, b-Ax_0
    Matx0 = multiplicacionMatrizVector(Matriz, x_0)
    print("Matx0", Matx0)

    r_0 = np.array(b) - np.array(Matx0)
    print("r_0", r_0)

    #Establecemos el máximo de iteraciones
    num_columnas = max_it
    print("num_columnas", num_columnas)

    # Generamos una matriz V y una h con todos sus valores a 0
    V = np.zeros((N, num_columnas+1))
    print("V", V)
    h = np.zeros((num_columnas+1, num_columnas))
    print("h", h)

    # Establecemos el v_1 al vector inicial normalizado.
    r_0_norm = np.linalg.norm(r_0, ord=2)
    print("r_0_norm", r_0_norm)
    V[:, 0] = np.array(r_0 / r_0_norm)
    print("V[:, 0]", V[:, 0])

    # Inicializamos el vector g
    g = np.zeros(num_columnas+1)
    g[0] = r_0_norm
    print("g", g)

    # Guardamos la norma para no repetir la operación en cada bucle
    b_norm = np.linalg.norm(b, ord=2)
    print("b_norm", b_norm)

    # Vector solucion
    x = np.zeros(N)
    print("x", x)

    N = N-1
    num_columnas = num_columnas - 1
    print("N", N)
    print("num_columnas", num_columnas)

    n=0
    while n<=(num_columnas):
        print("n", n)
        t = multiplicacionMatrizVector(Matriz, V[:,n])
        print("t", t)
        i=0
        while i <= n:    
            print("i", i)
            h[i][n] = multiplicacionDosVectores(V[:,i], t)
            print("h[i][n]", h[i][n])
            aux = multiplicacionValorVector(h[i][n], V[:,i])
            print("aux", aux)
            t = [t[k] - aux[k] for k in range(min(len(t), len(aux)))]
            print("t", t)
            i+=1
        t_norm = np.linalg.norm(t, ord=2)
        print("t_norm", t_norm)
        h[n+1][n] = t_norm
        print("h[n+1][n]", h[n+1][n])
        V[:,n+1] = t / t_norm
        print("V[:,n+1]", V[:,n+1])

        j=0
        while j<=n-1:
            print("j", j)
            c_j = abs(h[j][j]) / (np.sqrt( h[j][j]*h[j][j] + h[j+1][j]*h[j+1][j]  ))
            print("c_j", c_j)
            s_j = (h[j+1][j] / h[j][j])*c_j
            print("s_j", s_j)
            h[j][n] = c_j*h[j][n] + s_j*h[j+1][n]
            print("h[j][n]", h[j][n])
            h[j+1][n] = -s_j*h[j][n] + c_j*h[j+1][n]
            print("h[j+1][n]", h[j+1][n])
            j += 1
        c_n = abs(h[n][n]) / (np.sqrt( h[n][n]*h[n][n] + h[n+1][n]*h[n+1][n]  ))
        print("c_n", c_n)
        s_n = (h[n+1][n] / h[n][n])*c_n
        print("s_n", s_n)
        h[n][n] = c_n*h[n][n] + s_n*h[n+1][n]
        print("h[n][n]", h[n][n])
        h[n+1][n] = 0
        print("h[n+1][n]", h[n+1][n])

        g[n] = c_n*g[n]
        print("g[n]", g[n])
        g[n+1] = -s_n*g[n]
        print("g[n+1]", g[n+1])

        conver = abs(g[n+1])/b_norm
        print("conver", conver)
        if conver <= tol and n>0:
            h_reducida = h[:n, :n]
            print("h_reducida", h_reducida)
            v_reducida = V[:, :n]
            print("v_reducida", v_reducida)
            g_reducido = g[:n]
            print("g_reducido", g_reducido)
            inversa = np.linalg.inv(h_reducida)
            print("inversa", inversa)
            VnH_1n = multiplicacionDosMatrices(v_reducida, inversa)
            print("VnH_1n", VnH_1n)
            x = r_0 + multiplicacionMatrizVector(VnH_1n, g_reducido)
            print("x", x)
        n +=1
    
    print(multiplicacionMatrizVector(modificarMatriz(A, 0.85), x))
    return x



def pagerank_gmres(A, beta=0.85, tol=1e-6):
    n = A.shape[0]
    M = (beta * A) + ((1 - beta) / n) * np.ones((n, n))
    b = np.ones(n) / n
    x, _ = gmres(M, b, rtol=tol)
    return x / np.sum(x)



if __name__ == "__main__":

    # print("Creando matriz")
    # A = matrizPageRank(5)
    # print("matriz creada")
    A = np.array([[1/2, 1/3, 0, 0],
                  [0, 1/3, 0, 1],
                  [0, 1/3, 1/2, 0],
                  [1/2, 0, 1/2, 0]])
    alpha = 0.85
    

    x = GMRES(A, alpha, 5, 0.00000000001)
    print("Vector solución", x)


    # Calculamos el PageRank utilizando GMRES
    pagerank = pagerank_gmres(A)
    print("PageRank:", pagerank)