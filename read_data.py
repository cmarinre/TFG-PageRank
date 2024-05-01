
import numpy as np


def read_data_Andrew(file_path):
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
        # Obtenemos la primera linea de ese nodo, que obtiene el numero del nodo y el número de links que recibe
        # El map nos hace que todo lo que obtengamos lo convierta en entero
        node, _, _ = map(int, data[0].split())

        # Creamos una lista para almacenar los enlaces de este nodo
        links = []

        # Iteramos hasta encontrar el próximo nodo
        for line in data[1:]:
            values = list(map(int, line.split()))
            # Hemos llegado al próximo nodo
            if values[1] != node:
                break  
            # Añadimos el enlace sin el último valor (-1)
            links.append(values[:-1])
        
        # Para cada pareja de links que tenemos en el array
        for link in links:
            # Guardamos en la matriz los datos el link y partido por el núm de links, así ya va normalizado.
            matrix[link[0] - 1, node - 1] = 1 / len(links)

        # Eliminamos las líneas correspondientes al nodo procesado
        del data[:len(links) + 1]

    return matrix

def read_data_stanford(file_path):
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
            matrix[link[0] - 1, node - 1] = 1 / len(links)

        # Eliminamos las líneas correspondientes al nodo procesado
        del data[:len(links)]

    return matrix

if __name__ == "__main__":
    # A = read_data_Andrew("./datos/Andrew.mtx")
    A = read_data_stanford("./datos/prueba_stanford.mtx")
    print(A)
