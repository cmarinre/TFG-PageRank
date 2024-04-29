import numpy as np
from funciones_comunes import multiplicacionMatrizVector

# Método de las potencias estándar, calculando la convergencia con la norma 1
# y multiplicando la matriz con nuestra función definida arriba
def power_method(matrix, vector, max_iterations, tolerance):

    # Ejecutamos el método de las potencias y paramos cuando el número de iteraciones sea el máximo
    # O cuando cumple el factor de tolerancia
    for i in range(max_iterations):

        # Multiplicación de la matriz por el vector
        new_vector = multiplicacionMatrizVector(matrix, vector)
        
        # Aquí deberíamos dividir por la norma pero la norma siempre es 1.
        # new_vector = new_vector / np.linalg.norm(new_vector, ord=1)

        # Comprobación de convergencia
        resta = [new_vector[i] - vector[i] for i in range(min(len(new_vector), len(vector)))]
        if np.linalg.norm(resta, ord=1) < tolerance:
            break

        # Guardamos el vector nuevo
        vector = new_vector

    return vector, i

# Método de las potencias estándar, calculando la convergencia como la convergencia de las componentes una a una
# y multiplicando la matriz con nuestra función definida arriba
def power_method_convergence(matrix, vector, max_iterations, tolerance):

    # Ejecutamos el método de las potencias y paramos cuando el número de iteraciones sea el máximo
    # O cuando cumple el factor de tolerancia
    for i in range(max_iterations):
        # Multiplicación de la matriz por el vector
        matrix_vector_product = multiplicacionMatrizVector(matrix, vector)

        #Por si la norma es 0
        if(np.linalg.norm(matrix_vector_product, ord=1)==0): 
            print("No se puede dividir por 0")
            break

        # Cálculo del nuevo vector
        new_vector = matrix_vector_product / np.linalg.norm(matrix_vector_product, ord=1)

        # Comprobación de convergencia
        resta = [abs(new_vector[i] - vector[i]) for i in range(min(len(new_vector), len(vector)))]
        if all(abs(valor) < tolerance for valor in resta):
            break

        # Guardamos el vector nuevo
        vector = new_vector

    return vector, i


if __name__ == "__main__":
    print("Aquí no hay código. Vaya a comparacion_powers.py, por favor.")