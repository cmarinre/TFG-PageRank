
import numpy as np


def read_data(file_path):
    # Leemos el archivo
    with open(file_path, 'r') as file:
        data = file.readlines()

    # Obtenemos el número de nodos
    num_nodes, _, _ = map(int, data[0].split())

    # Eliminamos esa fila
    data = data[1:]
    
    # Inicializamos la matriz con ceros
    matrix = np.zeros((num_nodes, num_nodes))

   # Para cada nodo
    while data:
        # Obtenemos la primera linea de ese nodo y cogemos el nodo 
        _, node = map(int, data[0].split())

        # Creamos una lista para almacenar los enlaces de este nodo
        links = []

        # Iteramos hasta encontrar el próximo nodo
        for line in data:
            values = list(map(int, line.split()))
            # Hemos llegado al próximo nodo
            if values[1] != node:
                break  
            # Añadimos el enlace
            links.append(values)
        
        # Para cada pareja de links que tenemos en el array
        for link in links:
            # Guardamos en la matriz los datos el link y partido por el núm de links, así ya va normalizado.
            matrix[link[0] - 1,node - 1] = 1 / len(links)

        # Eliminamos las líneas correspondientes al nodo procesado
        del data[:len(links)]

    return matrix

def read_data_hollins(file_path):
    # Leemos el archivo
    with open(file_path, 'r') as file:
        data = file.readlines()

    # Obtenemos el número de nodos
    num_nodes, _, _ = map(int, data[0].split())

    # Eliminamos esa fila
    data = data[1:]
    
    # Inicializamos la matriz con ceros
    matrix = np.zeros((num_nodes, num_nodes))

   # Para cada nodo
    while data:
        # Obtenemos la primera linea de ese nodo y cogemos el nodo 
        _, node = map(int, data[0].split())

        # Creamos una lista para almacenar los enlaces de este nodo
        links = []

        # Iteramos hasta encontrar el próximo nodo
        for line in data:
            values = list(map(int, line.split()))
            # Hemos llegado al próximo nodo
            if values[1] != node:
                break  
            # Añadimos el enlace
            links.append(values)
        
        # Para cada pareja de links que tenemos en el array
        for link in links:
            # Guardamos en la matriz los datos el link y partido por el núm de links, así ya va normalizado.
            matrix[node - 1, link[0] - 1] = 1

        # Eliminamos las líneas correspondientes al nodo procesado
        del data[:len(links)]

    col_sums = matrix.sum(axis=0)
    for i in range(num_nodes):
        if col_sums[i] != 0:  # Evitamos la división por cero
            matrix[:, i] /= col_sums[i]

    return matrix

def read_data_minnesota(file_path):
    # Leemos el archivo
    with open(file_path, 'r') as file:
        data = file.readlines()

    # Obtenemos el número de nodos
    num_nodes, _, _ = map(int, data[0].split())

    # Eliminamos esa fila
    data = data[1:]
    
    # Inicializamos la matriz con ceros
    matrix = np.zeros((num_nodes, num_nodes))

   # Para cada nodo
    while data:
        # Obtenemos la primera linea de ese nodo y cogemos el nodo 
        _, node = map(int, data[0].split())

        # Creamos una lista para almacenar los enlaces de este nodo
        links = []

        # Iteramos hasta encontrar el próximo nodo
        for line in data:
            values = list(map(int, line.split()))
            # Hemos llegado al próximo nodo
            if values[1] != node:
                break  
            # Añadimos el enlace
            links.append(values)
        
        # Para cada pareja de links que tenemos en el array
        for link in links:
            # Guardamos en la matriz los datos el link y partido por el núm de links, así ya va normalizado.
            i, j = link[0] - 1, node - 1
            matrix[i, j] = 1 
            # Añadimos también el enlace inverso para garantizar la simetría
            matrix[j, i] = 1 
            

        # Eliminamos las líneas correspondientes al nodo procesado
        del data[:len(links)]

    col_sums = matrix.sum(axis=0)
    for i in range(num_nodes):
        if col_sums[i] != 0:  # Evitamos la división por cero
            matrix[:, i] /= col_sums[i]

    return matrix


def read_data_cz1268(file_path):
   # Leemos el archivo
    with open(file_path, 'r') as file:
        data = file.readlines()

    # Obtenemos el número de nodos
    num_nodes, _, _ = map(int, data[0].split())

    # Eliminamos esa fila
    data = data[1:]
    
    # Inicializamos la matriz con ceros
    matrix = np.zeros((num_nodes, num_nodes))

    # Para cada nodo
    while data:
        # Obtenemos la primera linea de ese nodo y cogemos el nodo 
        _, node, _ = map(float, data[0].split())
        node = int(node)

        # Creamos una lista para almacenar los enlaces de este nodo
        links = []

        # Iteramos hasta encontrar el próximo nodo
        for line in data:
            values = line.split()
            
            # Hemos llegado al próximo nodo
            if int(values[1]) != node:
                break  
            # Añadimos el enlace (omitiendo el tercer valor)
            links.append((int(values[0]), int(values[1])))

        # Para cada pareja de links que tenemos en el array
        for link in links:
            # Guardamos en la matriz los datos el link y partido por el núm de links, así ya va normalizado.
            matrix[link[0] - 1, node - 1] = 1 / len(links)

        # Eliminamos las líneas correspondientes al nodo procesado
        del data[:len(links)]

    return matrix

if __name__ == "__main__":
    A = read_data("./datos/prueba_stanford.mtx")
    print(A)
