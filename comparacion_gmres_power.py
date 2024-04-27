from comparacion_gmres import comparacion_gmres
from comparacion_powers import comparacionPowersMult
from funciones_comunes import matrizPageRank, modificarMatriz


if __name__ == "__main__":

    print("Creando matriz")
    A = matrizPageRank(5000)
    print("matriz creada")

    comparacion_gmres(A)
    
    print("Modificando matriz")
    M = modificarMatriz(A, 0.85)
    print("Matriz modificada")

    comparacionPowersMult(M)
