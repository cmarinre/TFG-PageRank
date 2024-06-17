import numpy as np

from read_data import read_data, read_data_cz1268


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
    S = np.ones((n,n))/n
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

def obtenerSolucionesNumpy(P, alphas):
    
    vector_solucion_python = np.zeros((len(alphas), len(P)))
    i=0
    for alpha in alphas:
        M = modificarMatriz(P, alpha)
        vector_solucion_python[i] = obtenerSolucionPython(M)
        i+=1

    return vector_solucion_python


def obtenerComparacionesNumpySoluciones(x, soluciones):
    normas = np.zeros(len(soluciones))
    i=0
    while i < len(soluciones):
        normas[i] = residuoDosVectores(soluciones[i], x[i])
        i+=1

    return normas

def obtenerComparacionesNumpy(P, alphas, x):
    
    normas = np.zeros(len(alphas))
    i=0
    for alpha in alphas:
        M = modificarMatriz(P, alpha)
        vector_solucion_python = obtenerSolucionPython(M)
        print("vector python", vector_solucion_python)
        normas[i] = residuoDosVectores(vector_solucion_python, x[i])
        i+=1

    return normas

def residuoDosVectores(x1, x2):

    if(x1[0]<0): x1 = np.dot(-1, x1)
    if(x2[0]<0): x2 = np.dot(-1, x2)
    resta = x1-x2
    diferencia = np.linalg.norm(resta, ord=2)

    return diferencia


def guardar_diferencias_txt(diferencias, filename):
    with open(filename, 'w') as f:
        for tiempo, diferencia in diferencias:
            tiempo_str = str(tiempo).replace('.', ',')
            diferencia_str = str(diferencia).replace('.', ',')
            f.write(f"{tiempo_str};{diferencia_str}\n")

def comprobarVectorNumpy():
    P = read_data("./datos/minnesota2642.mtx")
    # P = read_data("./datos/hollins6012.mtx")
    # P = read_data("./datos/stanford9914.mtx")
    P = arreglarNodosColgantes(P)
    M = modificarMatriz(P, 0.85)
    vector_propio = obtenerSolucionPython(M)
    vector_propio = vector_propio / np.linalg.norm(vector_propio, ord=1)
    x_2 = np.dot(M, vector_propio)
    x_2 = x_2 / np.linalg.norm(x_2, ord=1)
    dif = residuoDosVectores(vector_propio, x_2)
    print("Diferencia", dif)

def calcularNumCond():
    # P = read_data("./datos/minnesota2642.mtx")
    # P = read_data("./datos/hollins6012.mtx")
    # P = read_data("./datos/stanford9914.mtx")
    P = read_data_cz1268("./datos/cz1268.mtx")

    # Calcular el número de condición
    cond_number = np.linalg.cond(P)
    print("Número de condición:", cond_number)


if __name__ == "__main__":
    calcularNumCond()