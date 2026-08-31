import csv
from collections import defaultdict

from ruteo import buscar_ruta_optima

with open('tablas/nodos.csv', mode='r', encoding='utf-8') as file:
    reader = csv.reader(file)

    coordenadas = {}
    next(reader)  # Skip the header row
    for fila in reader:
        coordenadas[fila[0]] = (float(fila[1]), float(fila[2]))

with open('tablas/arista.csv', mode= 'r', encoding='utf-8') as file2:
    reader = csv.reader(file2)
    next(reader)  # Skip the header row
    vecinos = defaultdict(list)

    for fila in reader:
        origen = fila[0]
        destino = fila[1]
        distancia = float(fila[2])
        nodo1, nodo2 = fila[0], fila[1]
        vecinos[origen].append((destino, distancia))
        vecinos[destino].append((origen, distancia))

        if origen not in vecinos:
             raise ValueError(f"La arista menciona '{origen}', que no existe en nodos.csv")
        if destino not in vecinos:
             raise ValueError(f"La arista menciona '{destino}', que no existe en nodos.csv")

    total = 0
    for nodo in vecinos:
            total = total + len(vecinos[nodo])

    for nodo in vecinos:
        if len(vecinos[nodo]) == 0:
                print("Nodo sin vecinos:", nodo)



por_visitar = ["N18"]
visitados = []

while len(por_visitar) > 0:
    nodo = por_visitar.pop()
    visitados.append(nodo)
    
    for vecino, distancia in vecinos[nodo]:
        if vecino not in visitados and vecino not in por_visitar:
            por_visitar.append(vecino)


print(buscar_ruta_optima(vecinos, "N10", "N15"))
print(buscar_ruta_optima(vecinos, "N14", "N17"))
print(buscar_ruta_optima(vecinos, "N11", "N31"))