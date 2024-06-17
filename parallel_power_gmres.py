

import time

import numpy as np

from funciones_comunes import (arreglarNodosColgantes,
                               obtenerComparacionesNumpySoluciones,
                               obtenerSolucionesNumpy)
from parallel_gmres import parallel_gmres
from parallel_power import paraller_power_modified_MedicionNumIt
from read_data import read_data, read_data_cz1268


def parallel_power_gmres(P, v, alphas, tol, max_it_p, max_it_g, m):    
    x, num_it, res, mv_p = paraller_power_modified_MedicionNumIt(P, v, max_it_p, tol, alphas)
    x, res, mv_g, numit = parallel_gmres(P, v, m, alphas, tol, max_it_g, x)
    return x, mv_p + mv_g



if __name__ == "__main__":

    P = read_data_cz1268("./datos/cz1268.mtx")
    # P = read_data("./datos/minnesota2642.mtx")
    # P = read_data("./datos/hollins6012.mtx")
    # P = read_data("./datos/stanford9914.mtx")
    P = arreglarNodosColgantes(P)

    # P = np.array([[1/2, 1/3, 0, 0],
    #               [0, 1/3, 0, 1],
    #               [0, 1/3, 1/2, 0],
    #               [1/2, 0, 1/2, 0]])
    
    N = len(P)
    v = np.ones(N)/N
    m = 2
    tol=1e-4
    max_it_p = 5
    max_it_g = 100
    alphas = np.zeros(50)
    for i in range(0,50):
        alphas[i] = (i+50)*0.01
    # alphas = np.array([0.5, 0.51])

    start_time = time.time()
    x, num_it_total = parallel_power_gmres(P, v, alphas, tol, max_it_p, max_it_g, m)
    end_time = time.time()
    elapsed_time = end_time - start_time


    print("El tiempo de ejecución del PARALLEL POWER fue de: {:.5f} segundos".format(elapsed_time))

    soluciones = obtenerSolucionesNumpy(P, alphas)
    normas = obtenerComparacionesNumpySoluciones(x, soluciones)
    
    print("Solucion python", soluciones)
    print("Vectores solución", x)

    print("normas", normas)
    print("Número de iteraciones", num_it_total)

