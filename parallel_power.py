import copy
import time

import numpy as np

from funciones_comunes import (arreglarNodosColgantes, guardar_numit,
                               modificarMatriz,
                               obtenerComparacionesNumpySoluciones,
                               obtenerSolucionesNumpy, obtenerSolucionPython,
                               residuoDosVectores)
from read_data import read_data, read_data_cz1268, read_data_minnesota


# Función que paraleliza la ejecución del método de las potencias para distintos alphas
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

        if res[i] >= tolerance:
            x[i] = r[i] + v


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


# Función que paraleliza la ejecución del método de las potencias para distintos alphas que mide el número de iteraciones
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



if __name__ == "__main__":

    # P = read_data_minnesota("./datos/minnesota2642.mtx")
    # P = read_data("./datos/hollins6012.mtx")
    P = read_data("./datos/stanford9914.mtx")
    P = arreglarNodosColgantes(P)

    
    N = len(P)
    v = np.ones(N)/N
    tol=1e-8
    max_it = 10000

    alphas = np.zeros(50)
    for i in range(0,50):
        alphas[i] = (i+50)*0.01
        

    start_time = time.time()
    x, num_it, res, num_it_total = paraller_power_modified_MedicionNumIt(P, v, max_it, tol, alphas)
    end_time = time.time()
    elapsed_time = end_time - start_time

    print("El tiempo de ejecución del PARALLEL POWER fue de: {:.5f} segundos".format(elapsed_time))


    print("Número de iteraciones", num_it_total)
    print("Número de iteraciones", num_it)
    guardar_numit(num_it, "numit.txt")

    print("Residuo", res)

