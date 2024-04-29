from funciones_comunes import multiplicacionMatrizVectorConCeros
import copy


#Método de las potencias adaptado, multiplcando la matriz con nuestra función especial
def adaptive_power_method(matrix, vector, max_iterations, tolerance):

    # Obtenemos la dimensión de la matriz
    n = len(matrix)

    # Vector de convergencia de componentes
    converg_comp = [0] * n

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
    print("Aquí no hay código. Vaya a comparacion_powers.py, por favor.")