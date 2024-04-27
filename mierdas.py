
def comprobacionArnoldi():
    # A = np.array([[1/2, 1/3, 0, 0],
    #             [0, 1/3, 0, 1],
    #             [0, 1/3, 1/2, 0],
    #             [1/2, 0, 1/2, 0]])
    print("Creando matriz")
    A = matrizPageRank(5)
    print("matriz creada")

    M = modificarMatriz(A, 0.85)
    
    N = len(M)
    num_cols = 4
    if(num_cols>N):
        print("Error con el número de columnas")
        return
    
    randomv = np.random.rand(N)
    V, H = arnoldi(M, randomv, num_cols)

    V = np.array(V)
    H = np.array(H)
    print("V", V)
    print("H", H)


    # Trasponemos V porque nos va a hacer falta traspuesta
    Vtrasp = np.transpose(V)

    # En efecto V cumple con lo que debe
    # I = multiplicacionDosMatrices(V, trasp_v)
    # print("I", I)

    # De V tenemos que quedarnos con los primeros n vectores y guardar en v_n+1 el ultimo:
    V_n = Vtrasp[:, :-1]  # Seleccionar todas las columnas excepto la última V_n
    v_n1 = Vtrasp[:, -1]  # Seleccionar la última columna v_n+1
    # print("V_n", V_n)
    # print("v_n1", v_n1)


    # Y de H tenemos que quitarle la última fila Y QUEDARNOS ADEMÁS CON EL VALOR SUELTO 
    H_n = H[:-1, :]  # Seleccionar todas las filas excepto la última
    h_n1 = H[num_cols-1][num_cols-2]
    print("H_n", H_n)
    print("h_n1", h_n1)

    MVn = multiplicacionDosMatrices(M, V_n)
    print("MVn", np.array(MVn))

    VNhN = multiplicacionDosMatrices(V_n, H_n)
    # print("VNhN", np.array(VNhN))

    auxiliar = np.zeros((N, num_cols-1))  # Crear una matriz de ceros de tamaño N x num_cols
    auxiliar[:, -1] = v_n1  # Asignar el vector v_n+1 a la última columna de la matriz
    auxiliar = multiplicacionValorMatriz(h_n1, auxiliar)
    # print("auxiliar", np.array(auxiliar))

    matriz_final =VNhN + auxiliar
    print("final", np.array(matriz_final))

    



def auxiliar(num_cols):
    randomv = np.random.rand(N)
    V, H = arnoldi(M, randomv, num_cols)

    # Trasponemos V porque nos va a hacer falta traspuesta
    Vtrasp = np.transpose(V)

    # De V tenemos que quedarnos con los primeros n vectores y guardar en v_n+1 el ultimo:
    V_n = Vtrasp[:, :-1]  # Seleccionar todas las columnas excepto la última V_n

    # Nuestro sistema es de la forma (I-alphaA)x = (1-alpha)v
    v = [1/N]*N
    Matriz = np.eye(N) - np.array(multiplicacionValorMatriz(alpha, A))
    b = multiplicacionValorVector(1-alpha, v)
    print(b)
    Mx0 = multiplicacionMatrizVector(Matriz, V[0])
    print(Mx0)
    r0 = np.linalg.norm(np.array(Mx0) - np.array(b), ord=2)
    print(r0)
    rotacionesGivens(np.array(H), r0,)


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