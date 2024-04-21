import numpy as np


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
    resultado = [0] * n
    
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

#Método de las potencias adaptado, multiplcando la matriz con nuestra función especial
def adaptive_power_method(matrix, max_iterations=50000, tolerance=0.000000000001):

    # Obtenemos la dimensión de la matriz
    n = len(matrix)

    # Vector de convergencia de componentes
    converg_comp = [0] * n

    # Generamos un vector aleatorio de tamaño n
    vector = np.random.rand(n)

    # Lo normalizamos dividiendo por la norma 1 (suma de las componentes)
    vector /= np.linalg.norm(vector, ord=1)

    # Inicializamos los dos vectores y la matriz
    x_k = vector
    x_kii = [0] * n
    matrix_Aii = matrix


    # En cada iteración
    for j in range(max_iterations):

        # Calculamos el nuevo vector. Primero mutiplicando A por el vector y luego sumandole el vector de las componentes que ya han convergido
        x_k1 = multiplicacionMatrizVectorConCeros(matrix_Aii, x_k, converg_comp)
        x_k1 = [x_k1[i] + x_kii[i] for i in range(min(len(x_k1), len(x_kii)))]

        # Comprobamos componente por componente si ha cumplido el criterio de convergencia.
        # En los que lo haya cumplido, la fila la ponemos a 0
        # Y en el nuevo vector ponemos el valor y ya no lo tocamos nunca más

        # Para cada componente
        for i in range(len(vector)):
            # Si la componente no había cumplido ya el criterio de convergencia
            if converg_comp[i]==0:
                # Si la cumple ahora
                resta = abs(x_k1[i] - x_k[i])
                if  resta < tolerance:
                    # Lo apuntamos en el vector de convergencia
                    converg_comp[i] = 1
                    # Ponemos en el x_kii a su valor
                    x_kii[i] = x_k1[i]
                    # Y en la matriz esa fila a 0
                    matrix_Aii[i] = [0] * len(matrix_Aii[i])
        

        # Comprobación de convergencia
        resta = [abs(x_k1[i] - x_kii[i]) for i in range(min(len(x_k1), len(x_kii)))]
        if all(abs(valor) < tolerance for valor in (resta)):
            break

        # Guardamos el vector nuevo
        x_k = x_k1

    return x_k, j


if __name__ == "__main__":
    print("Aquí no hay código. Vaya a funciones_comunes.py, por favor.")