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


# Función que dada una matriz A y un escalas alpha devuelve la matriz modificada para ser estocástica por columnas y con dim(V_1(A))=1
def modificarMatriz(A, alpha):

    n = len(A)
    S = [[1/n] * (n) for _ in range(n)]
    M = np.dot(alpha, A) + np.dot((1-alpha), S)
    return M

def arreglarNodosColgantes(A):
    n = len(A)
    nueva_matriz = np.zeros((n+1, n+1))  # Crear una matriz de (n+1)x(n+1) llena de ceros
    nueva_matriz[:n, :n] = A  # Copiar la matriz original en la esquina superior izquierda

    # Colocar 1 en el n+1xn+1
    nueva_matriz[n][n] = 1 
    
    # Poner 1s en todas las columnas adicionales a la columna nx
    for i in range(n):
        if np.all(A[:, i] == 0):  # Si todas las entradas de la columna son cero
            nueva_matriz[n][i] = 1  # Poner 1 en la nueva columna para esta fila

    return nueva_matriz


def obtenerSolucionPython(A):
    eigenvalues, eigenvectors = np.linalg.eig(A)

    # Encuentra el índice del valor propio que es igual a 1
    index = np.where(np.isclose(eigenvalues, 1))[0][0]

    # Obtiene el vector propio correspondiente
    eigenvector = eigenvectors[:, index]
    eigenvector = eigenvector / np.linalg.norm(eigenvector, ord=1)
    return eigenvector

def residuoDosVectores(x1, x2):
    if(x1[0]<0): x1 = np.dot(-1, x1)
    if(x2[0]<0): x2 = np.dot(-1, x2)
    resta = [abs(x1[i] - x2[i]) for i in range(len(x1))]
    diferencia = np.linalg.norm(resta, ord=1)
    return diferencia



