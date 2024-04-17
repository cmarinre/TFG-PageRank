import numpy as np

def matrizPageRank(n):
    # Definir una matriz de ceros
    matrix = np.zeros((n, n))

    # Asignar valores aleatorios a las conexiones entre páginas
    for i in range(n):
        # Determinar el número de enlaces salientes desde la página i
        num_outlinks = np.random.randint(5, 10)
        
        # Generar índices aleatorios para los enlaces salientes
        outlink_indices = np.random.choice(n, size=num_outlinks, replace=False)
        
        # Asignar probabilidades de enlace a los enlaces salientes
        for j in outlink_indices:
            matrix[j, i] = np.random.uniform(0.1, 0.9)

    # Normalizar las columnas para que la suma sea igual a 1
    matrix = matrix / matrix.sum(axis=0)
    return matrix


def generarMatrizAleatoria(n):
    # Generar una matriz aleatoria de dimensiones 1000x1000
    matriz = np.random.rand(n, n)

    # Normalizar cada columna para que la suma sea igual a 1
    matriz_estocastica = matriz / matriz.sum(axis=0)

    # Verificar que la matriz tenga el valor propio 1
    autovalores, _ = np.linalg.eig(matriz_estocastica)

    # Encontrar el autovalor dominante
    autovalor_maximo = np.max(autovalores)
    print("Max auto ",autovalor_maximo)

    return matriz
