import copy
import time

import numpy as np

from funciones_comunes import (arreglarNodosColgantes, modificarMatriz,
                               obtenerSolucionPython, residuoDosVectores)
from read_data import read_data


def paraller_power_modified(P, vector, max_mv, tolerance, alphas):
    # Obtenemos la dimensión de la matriz y el número de alphas que tenemos
    N = len(P)
    s = len(alphas)

    # Creamos los vectores iniciales vacíos
    x = np.zeros((s, N))
    r = np.ones((s, N))
    res = np.ones(s)
    num_it = np.zeros(s)
    v = vector

    # Calculamos u
    u = np.dot(P, v) - v
    print("u", u)
    # mv será la variable con la que mediremos el número de iteración en el que estamos
    mv=1

    # Para cada alpha
    for i in range(s):
        r[i] = np.dot(alphas[i],u)
        res[i] = np.linalg.norm(r[i], ord=1)
        print("r", r)

        if res[i] >= tolerance:
            x[i] = r[i] + v
            print("x", x)


    print("WHILE")

    while max(res) >= tolerance and mv <= max_mv:
        u = np.dot(P,u)
        mv += 1
        print("u", u)
        for i in range(s):
            if res[i]>=tolerance:
                r[i] = np.dot(alphas[i]**(mv+1), u)
                print("r", r)
                res[i] = np.linalg.norm(r[i], ord=1)
                print("res", res)
                if res[i]>= tolerance:
                    x[i] = r[i] + x[i]
                    print("x", x)
      

    return x, num_it, res, mv



def paraller_power_modified_MedicionNumIt(P, vector, max_mv, tolerance, alphas):
    # Obtenemos la dimensión de la matriz y el número de alphas que tenemos
    N = len(P)
    s = len(alphas)

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
        r[i] = np.dot(alphas[i],u)
        res[i] = np.linalg.norm(r[i], ord=1)
        if res[i] >= tolerance:
            x[i] = r[i] + v
        else:
            # Si ha cumplido el criterio de convergencia guardamos el número de iteraciones que ha tardado
            num_it[i] = mv

    while max(res) >= tolerance and mv <= max_mv:
        u = np.dot(P,u)
        mv += 1
        for i in range(s):
            if res[i]>=tolerance:
                r[i] = np.dot(alphas[i]**(mv), u)
                res[i] = np.linalg.norm(r[i], ord=1)
                if res[i]>= tolerance:
                    x[i] = r[i] + x[i]
                # Si ha cumplido el criterio de convergencia 
                else:
                    # Y no hemos guardado antes su núm it (está a 0) , guardamos el número de iteraciones
                    if num_it[i]==0: num_it[i] = mv

            # Si ha cumplido el criterio de convergencia guardamos el número de iteraciones que ha tardado
            else:
                if num_it[i]==0: num_it[i] = mv


    return x, num_it, res, mv

def obtenerSolucionesNumpy(P, alphas):
    
    vector_solucion_python = np.zeros((len(alphas), len(P)))
    i=0
    for alpha in alphas:
        M = modificarMatriz(P, alpha)
        vector_solucion_python[i] = obtenerSolucionPython(M)
        i+=1

    return vector_solucion_python

def obtenerComparacionesNumpy(P, alphas, x):
    
    normas = np.zeros(len(alphas))
    i=0
    for alpha in alphas:
        M = modificarMatriz(P, alpha)
        vector_solucion_python = obtenerSolucionPython(M)
        normas[i] = residuoDosVectores(vector_solucion_python, x[i])
        i+=1

    return normas

def obtenerComparacionesNumpySoluciones(x, soluciones):
    normas = np.zeros(len(soluciones))
    i=0
    while i < len(soluciones):
        normas[i] = residuoDosVectores(soluciones[i], x[i])
        i+=1

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
    tol=1e-6
    max_it = 100000

    alphas = np.zeros(50)
    for i in range(0,50):
        alphas[i] = (i+50)*0.01
        

    start_time = time.time()
    x, num_it, res, num_it_total = paraller_power_modified_MedicionNumIt(P, v, max_it, tol, alphas)
    end_time = time.time()
    elapsed_time = end_time - start_time

    print("El tiempo de ejecución del PARALLEL POWER fue de: {:.5f} segundos".format(elapsed_time))

    # soluciones = obtenerSolucionesNumpy(P, [alphas[49]])
    # normas = obtenerComparacionesNumpySoluciones([x[49]], soluciones)
    
    # print("Solucion python", soluciones)
    # print("Vectores solución", x)

    print("Número de iteraciones", num_it_total)
    print("Número de iteraciones", num_it)

    # print("Residuo", res)
    # print("Norma residual", normas)


