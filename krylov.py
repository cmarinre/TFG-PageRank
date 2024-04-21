
import numpy as np
from funciones_comunes import matrizPageRank


def arnoldi(A):
    N = len(A)
    v = [0]*N
    v[0] = np.random.rand(N)
    # v_2 = np.dot(A, v_1)
    # v_3 = np.dot(A, v_2)

    print("v_base", v)

    h = [[0]*N]*N
    
    for n in range(0, N):
        for i in range(0,n):
            print("n", n)
            print("vn", v[n])
            Avn = np.dot(A, v[n])
            print("Avn", Avn)
            h[i][n] = np.dot(v[n], Avn)
            print("h", h)
            vs = Avn - sum(np.dot(h[i][n], v[i]))
            print("vs", vs)
            vs_norm = np.linalg.norm(vs, ord=1)
            print("vs_norm", vs_norm)
            h[n+1][n] = vs_norm
            v[n+1] = vs / vs_norm
            print("v", v)


    return v,h



if __name__ == "__main__":

    print("Creando matriz")
    A = matrizPageRank(5)
    print("matriz creada", A)

    v, h = arnoldi(A)
    print(v)
    print(h)
