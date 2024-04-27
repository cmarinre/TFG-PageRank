
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


