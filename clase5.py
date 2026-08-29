import heapq

# 1. Definir el grafo usando un diccionario de adyacencia
# Cada nodo tiene una lista de tuplas (vecino, costo_arista)
grafo = {
    'A': [('B', 2), ('D', 5)],
    'B': [('A', 2), ('C', 2)],
    'C': [('B', 2), ('D', 2)],
    'D': [('C', 2), ('A', 5)]
}

def dijkstra(grafo, inicio, destino):
    # Diccionario para guardar la menor distancia conocida a cada nodo
    distancias = {nodo: float('inf') for nodo in grafo}
    distancias[inicio] = 0
    
    # Cola de prioridad (heap). Almacena tuplas: (distancia_acumulada, nodo)
    # heapq ordena las tuplas SIEMPRE por el primer elemento (la distancia)
    cola_prioridad = [(0, inicio)]
    
    while cola_prioridad:
        # Extrae el nodo con la distancia más corta actual
        distancia_actual, nodo_actual = heapq.heappop(cola_prioridad)
        
        # Si ya llegamos al destino, terminamos
        if nodo_actual == destino:
            return distancia_actual
            
        # Si encontramos una distancia mayor a la ya procesada, la ignoramos
        if distancia_actual > distancias[nodo_actual]:
            continue
            
        # Revisar los vecinos del nodo actual
        for vecino, peso_arista in grafo[nodo_actual]:
            distancia_nueva = distancia_actual + peso_arista
            
            # Si encontramos un camino más corto hacia el vecino, lo actualizamos
            if distancia_nueva < distancias[vecino]:
                distancias[vecino] = distancia_nueva
                heapq.heappush(cola_prioridad, (distancia_nueva, vecino))
                
    return distancias[destino]

# Ejecutar el algoritmo
distancia_final = dijkstra(grafo, 'A', 'D')
print(f"La distancia más corta de A a D es: {distancia_final} metros.")
