import numpy as np

# Función para la generación de una matriz dado un tamaño, que se parezca a una matriz de enlaces.
# Intentamos establecer el número de enlaces salientes de cada página y ponerlos aleatoriamente en páginas distintas.
def matrizPageRank(n):
    matrix = np.zeros((n, n))

    for i in range(n):
        # Determinar el número de enlaces salientes desde la página i
        num_outlinks = np.random.randint(1, n)  # Puede haber hasta n enlaces
        
        # Si no hay enlaces salientes, le ponemos en una aleatoria un 1
        if num_outlinks == 0:
            random_page = np.random.randint(n)
            matrix[random_page, i] = 1
        else:
            # Asignar probabilidades de enlace iguales a los enlaces salientes
            probabilidad_enlace = 1 / num_outlinks
            outlink_indices = np.random.choice(n, size=num_outlinks, replace=False)
            for j in outlink_indices:
                matrix[j, i] = probabilidad_enlace

    # Normalizar las columnas para que la suma sea igual a 1
    matrix = matrix / np.sum(matrix, axis=0, keepdims=True)
    
    return matrix

# Función para multiplicar dos matrices.
def multiplicacionDosMatrices(A, B):

    #Comprobamos que se puedan multiplicar
    if len(A[0]) != len(B):
        print("No se puede multiplicar estas matrices.")
        return None

    # Inicializamos el vector resultado 
    resultado = np.zeros((len(A), len(B[0])))

    # Multiplicamos las matrices
    for i in range(len(A)):
        for j in range(len(B[0])):
            for k in range(len(B)):
                resultado[i][j] += A[i][k] * B[k][j]

    return resultado

# Función para multiplicar una matriz y un vector.
def multiplicacionMatrizVector(A, v):
    N, n = len(A), len(A[0])

    # Comprobamos que se puedan multiplicar
    if n != len(v):
        print("No se puede multiplicar esta matriz y este vector.")
        return None
    
    # Inicializamos el vector resultado 
    resultado = np.zeros(N)
    
    # Multiplicamos la matriz por el vector
    for i in range(N):
        for j in range(n):
            resultado[i] += A[i][j] * v[j]
    
    return resultado

# Función para multiplicar una matriz y un vector sabiendo que algunas filas de la matriz están completamente a cero.
# A es la matriz, v el vector y ceros el vector que tiene un 0 en las filas de A normales y un 1 en la posición de las filas que están a 0
def multiplicacionMatrizVectorConCeros(A, v, ceros):

    #Guardamos la dim de la matriz
    n = len(A)

    #Comprobamos que se puedan multiplicar
    if n != len(v):
        print("No se puede multiplicar esta matriz y este vector.")
        return None
    
    # Inicializamos el vector resultado 
    resultado = np.zeros(n)
    
    # Multiplicamos la matriz por el vector si la fila del vector no está completa a 0
    for i in range(n):
        # Si la fila está completa a 0, ponemos ese valor directamente a 0
        if ceros[i]!=0:
            resultado[i] = 0
        #Si no, multiplicamos normal
        else:
            for j in range(n):
                resultado[i] += A[i][j] * v[j]
    return resultado

# Función para multiplicar dos vectores.
def multiplicacionDosVectores(V1, V2):
    #Guardamos la dim de la matriz
    n = len(V1)

    #Comprobamos que se puedan multiplicar
    if n != len(V2):
        print("No se puede multiplicar esta matriz y este vector.")
        return None
    
    # Inicializamos el vector resultado 
    resultado = 0
    
    # Multiplicamos la matriz por el vector
    for i in range(n):
        resultado += V1[i] * V2[i]
    
    return resultado
    
# Función para multiplicar un valor por un vector.
def multiplicacionValorVector(k, v):

    n = len(v)
    # Inicializamos el vector resultado 
    resultado = np.zeros(n)
    # Multiplicamos la matriz por el vector
    for i in range(n):
        resultado[i] = k * v[i]

    return resultado

# Función para multiplicar una matriz cuadrada por un valor.
def multiplicacionValorMatriz(k, A):
    filas = len(A)
    columnas = len(A[0])
    for i in range(filas):
        for j in range(columnas):
            A[i][j] = A[i][j]*k

    return A

# Función para sumar dos matrices.
def sumaDosMatrices(A, B):
    filas = len(A)
    columnas = len(A[0])
    for i in range(filas):
        for j in range(columnas):
            A[i][j] = A[i][j] + B[i][j]
    return A

    
def modificarMatriz(A, alpha):

    n = len(A)
    S = [[1/n] * (n) for _ in range(n)]
    M = multiplicacionValorMatriz(alpha, A) + multiplicacionValorMatriz((1-alpha), S)
    return M