import numpy as np

# Función para multiplicar una matriz y un vector.
def multiplicacionMatrizVector(A, v):
    #Guardamos la dim de la matriz
    n = len(A)

    #Comprobamos que se puedan multiplicar
    if n != len(v):
        print("No se puede multiplicar esta matriz y este vector.")
        return None
    
    # Inicializamos el vector resultado 
    resultado = [0] * n
    
    # Multiplicamos la matriz por el vector
    for i in range(n):
        for j in range(n):
            resultado[i] += A[i][j] * v[j]
    
    return resultado

# Método de las potencias estándar, calculando la convergencia con la norma 1
# y multiplicando la matriz con nuestra función definida arriba
def power_method(matrix, max_iterations=50000, tolerance=0.000000000001):

    # Obtenemos la dimensión de la matriz
    n = len(matrix)

    # Generamos un vector aleatorio de tamaño n
    vector = np.random.rand(n)

    # Lo normalizamos dividiendo por la norma 1 (suma de las componentes)
    vector /= np.linalg.norm(vector, ord=1)


    # Ejecutamos el método de las potencias y paramos cuando el número de iteraciones sea el máximo
    # O cuando cumple el factor de tolerancia
    for i in range(max_iterations):
        # Multiplicación de la matriz por el vector
        matrix_vector_product = multiplicacionMatrizVector(matrix, vector)
        
        # Cálculo del nuevo vector
        new_vector = matrix_vector_product / np.linalg.norm(matrix_vector_product, ord=1)

        # Comprobación de convergencia
        if np.linalg.norm(new_vector - vector, ord=1) < tolerance:
            break

        # Guardamos el vector nuevo
        vector = new_vector

    return vector, i

# Método de las potencias estándar, calculando la convergencia como la convergencia de las componentes una a una
# y multiplicando la matriz con nuestra función definida arriba
def power_method_convergence(matrix, max_iterations=50000, tolerance=0.000000000001):

    # Obtenemos la dimensión de la matriz
    n = len(matrix)

    # Generamos un vector aleatorio de tamaño n
    vector = np.random.rand(n)

    # Lo normalizamos dividiendo por la norma 1 (suma de las componentes)
    vector /= np.linalg.norm(vector, ord=1)

    # Ejecutamos el método de las potencias y paramos cuando el número de iteraciones sea el máximo
    # O cuando cumple el factor de tolerancia
    for i in range(max_iterations):
        # Multiplicación de la matriz por el vector
        matrix_vector_product = multiplicacionMatrizVector(matrix, vector)
        
        # Cálculo del nuevo vector
        new_vector = matrix_vector_product / np.linalg.norm(matrix_vector_product, ord=1)

        # Comprobación de convergencia
        resta = [abs(new_vector[i] - vector[i]) for i in range(min(len(new_vector), len(vector)))]
        if all(abs(valor) < tolerance for valor in resta):
            break

        # Guardamos el vector nuevo
        vector = new_vector

    return vector, i


# Ejemplo de uso 
if __name__ == "__main__":
    print("Aquí no hay código. Vaya a funciones_comunes.py, por favor.")