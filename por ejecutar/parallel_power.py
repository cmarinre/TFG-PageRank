import copy
import time

import numpy as np

from funciones_comunes import arreglarNodosColgantes
from read_data import read_data


def paraller_power(P, vector, max_iterations, tolerance, alphas):
    k=0
    u1=[]
    u2=[]
    s = len(alphas)
    N = len(P)
    x_k = np.zeros((s, N))
    x_k_1 = np.zeros((s, N))
    salir = False
    while k < max_iterations and salir==False:
        if k==0:
            u1 = np.dot(P, vector)
            u2 = vector
            for i in range(s):
                x_k[i] = np.dot(alphas[i], u1) + np.dot((1-alphas[i]), u2)
        else:
            for i in range(s):
                x_k[i] = x_k_1[i] - np.dot(alphas[i], u1)
            
            u1 = np.dot(P, u1)
            u2 = np.dot(P, u2)

            for i in range(s):
                x_k[i] = np.dot(alphas[i], u1) + x_k[i] + np.dot((1-alphas[i])*alphas[i], u2)
        
        for i in range(s):
            x_k[i] = x_k[i] / np.linalg.norm(x_k[i], ord=1)

        for i in range(s):
            resta = [abs(x_k[i][j] - x_k_1[i][j]) for j in range(N)]
            if all(abs(valor) < tolerance for valor in resta):
                salir = True

        x_k_1 = copy.deepcopy(x_k)
        k+=1

    return x_k, k


def paraller_power_modified(P, vector, max_mv, tolerance, alphas):
    # Obtenemos la dimensión de la matriz y el número de alphas que tenemos
    s = len(alphas)
    N = len(P)

    # Creamos los vectores iniciales vacíos
    x = np.zeros((s, N))
    r = np.ones((s, N))
    res = np.ones(s)
    num_it = np.zeros(s)
    v = vector

    # Calculamos u
    u = np.dot(P, v) - v

    # mv será la variable con la que mediremos el número de iteración en el que estamos
    mv=1

    # Para cada alpha
    for i in range(s):
        r[i] = alphas[i]*u
        res[i] = np.linalg.norm(r[i], ord=2)
        if res[i]>= tolerance:
            x[i] = r[i] + v
        else:
            # Si ha cumplido el criterio de convergencia guardamos el número de iteraciones que ha tardado
            num_it[i] = mv

    while max(res) >= tolerance and mv <= max_mv:
        u = np.dot(P,u)
        mv += 1
        
        for i in range(s):
            if res[i]>=tolerance:
                r[i] = alphas[i]*u
                res[i] = np.linalg.norm(r[i], ord=2)
                if res[i]>= tolerance:
                    x[i] = r[i] + x[i]
                # Si ha cumplido el criterio de convergencia 
                else:
                    # Y no hemos guardado antes su núm it (está a 0) , guardamos el número de iteraciones
                    if num_it[i]==0: num_it[i] = mv

            # Si ha cumplido el criterio de convergencia guardamos el número de iteraciones que ha tardado
            else:
                if num_it[i]==0: num_it[i] = mv
            
            # Para que nuestros vectores estén siempre normalizados
            x[i] = x[i] / np.linalg.norm(x[i], ord=1)

        if((mv%100)==0): 
            print(mv)
            print(num_it)
    return x, num_it, res


def obtenerComparacionesNumpy(P, alphas, x):
    N = len(P)
    S = np.ones((N, N))/N
    M = np.dot(0.85, P) + np.dot(0.15, S)

    normas = np.zeros(len(alphas))
    # Calcular los valores y vectores propios
    valores_propios, vectores_propios = np.linalg.eig(M)

    # Encontrar el índice del valor propio más cercano a 1
    indice = np.argmin(np.abs(valores_propios - 1))

    # Obtener el vector propio asociado al valor propio 1
    vector_propio = vectores_propios[:, indice]
    vector_propio = vector_propio/np.linalg.norm(vector_propio, ord=1) 

    for i in range(len(alphas)):
        resta = np.array(x[i]) - np.array(vector_propio)
        normas[i] = np.linalg.norm(resta)
    return normas



if __name__ == "__main__":
    # P = read_data("./datos/minnesota2642.mtx")
    # P = read_data("./datos/hollins6012.mtx")
    P = read_data("./datos/stanford9914.mtx")
    P = arreglarNodosColgantes(P)

    # P = np.array([[1/2, 1/3, 0, 0],
    #               [0, 1/3, 0, 1],
    #               [0, 1/3, 1/2, 0],
    #               [1/2, 0, 1/2, 0]])
    
    N = len(P)
    v = np.ones(N)/N
    tol=1e-4
    max_it = 100000

    alphas = np.zeros(99)
    for i in range(99):
        alphas[i] = (i+1)*0.01

    # alphas = np.array([0.2, 0.4, 0.5, 0.6, 0.85, 0.9])
    print(alphas)

    start_time = time.time()
    x, num_it, res = paraller_power_modified(P, v, max_it, tol, alphas)
    end_time = time.time()
    elapsed_time = end_time - start_time

    print("El tiempo de ejecución del parallel power fue de: {:.5f} segundos".format(elapsed_time))

    # normas = obtenerComparacionesNumpy(P, alphas, x)

    print("Vectores solución", x)
    print("Número de iteraciones", num_it)
    print("Residuo", res)
    # print("Normas residuales", normas)

