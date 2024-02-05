import numpy as np

def power_method(matrix, max_iterations=100, tolerance=1e-6):
    # Inicialización de vector aleatorio
    n = len(matrix)
    vector = np.random.rand(n)
    vector /= np.linalg.norm(vector)

    # Iteraciones del método de las potencias
    for i in range(max_iterations):
        # Multiplicación de la matriz por el vector
        matrix_vector_product = np.dot(matrix, vector)

        # Cálculo del nuevo vector
        new_vector = matrix_vector_product / np.linalg.norm(matrix_vector_product)

        # Comprobación de convergencia
        if np.linalg.norm(new_vector - vector) < tolerance:
            break

        vector = new_vector

    # Cálculo del eigenvalor dominante
    eigenvalue = np.dot(np.dot(vector, matrix), vector)

    return eigenvalue, vector

# Ejemplo de uso
if __name__ == "__main__":
    # Definición de una matriz de ejemplo
    A = np.array([[4, -1],
                  [2,  1]])

    # Aplicación del método de las potencias
    eigenvalue, eigenvector = power_method(A)

    print("Eigenvalor dominante:", eigenvalue)
    print("Eigenvector correspondiente:", eigenvector)