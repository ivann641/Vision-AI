import numpy as np

def ordenar_esquinas(pts):
    pts = np.array(pts, dtype=np.float32).reshape(4, 2)

    suma = pts.sum(axis=1)             # x + y de cada punto
    resta = np.diff(pts, axis=1).ravel()  # y - x de cada punto

    ordenado = np.zeros((4, 2), dtype=np.float32)
    ordenado[0] = pts[np.argmin(suma)]   # 1: arriba-izquierda
    ordenado[1] = pts[np.argmin(resta)]  # 2: arriba-derecha
    ordenado[2] = pts[np.argmax(suma)]   # 3: abajo-derecha
    ordenado[3] = pts[np.argmax(resta)]  # 4: abajo-izquierda
    return ordenado


# Pruebas
trapecio = [(310, 200), (120, 40), (110, 180), (300, 60)]
mezclado = [(110, 180), (310, 200), (300, 60), (120, 40)]
rectangulo = [(200, 100), (50, 20), (200, 20), (50, 100)]

print(ordenar_esquinas(trapecio))
print(ordenar_esquinas(mezclado))
print(ordenar_esquinas(rectangulo))