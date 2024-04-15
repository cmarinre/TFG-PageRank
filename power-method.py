import numpy as np

def power_method(matrix, max_iterations=100, tolerance=1e-6):
    # Obtenemos la dimensión de la matriz
    n = len(matrix)
    # Generamos un vector aleatorio de tamaño n
    vector = np.random.rand(n)
    print(vector)
    vector = [1, 0, 0]
    # Lo normalizamos dividiendo por la norma 1 (suma de las componentes)
    vector /= np.linalg.norm(vector)


    # Ejecutamos el método de las potencias y paramos cuando el número de iteraciones sea el máximo
    # O cuando cumple el factor de tolerancia
    for i in range(max_iterations):
        # Multiplicación de la matriz por el vector
        matrix_vector_product = np.dot(matrix, vector)

        # Cálculo del nuevo vector
        new_vector = matrix_vector_product / np.linalg.norm(matrix_vector_product)

        # Comprobación de convergencia
        if np.linalg.norm(new_vector - vector) < tolerance:
            break

        # Guardamos el vector nuevo
        vector = new_vector

    return vector

# Ejemplo de uso
if __name__ == "__main__":
    # Definición de una matriz de ejemplo
    A = np.array([[1/2, 1/3, 0],
                  [1/2, 1/3, 0],
                  [0, 1/3, 1]])

    # Aplicación del método de las potencias
    eigenvector = power_method(A)

    print("Vector propio correspondiente:", eigenvector)
    print("Suma de los valores del vector propio para asegurar que su norma 1 es igual a 1:", np.sum(eigenvector))
