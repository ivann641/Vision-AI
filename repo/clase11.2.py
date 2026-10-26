import cv2
import numpy as np

def ordenar_esquinas(pts):
    pts = np.array(pts, dtype=np.float32).reshape(4, 2)

    suma = pts.sum(axis=1)                 # x + y de cada punto
    resta = np.diff(pts, axis=1).ravel()   # y - x de cada punto

    ordenado = np.zeros((4, 2), dtype=np.float32)
    ordenado[0] = pts[np.argmin(suma)]     # 1: arriba-izquierda
    ordenado[1] = pts[np.argmin(resta)]    # 2: arriba-derecha
    ordenado[2] = pts[np.argmax(suma)]     # 3: abajo-derecha
    ordenado[3] = pts[np.argmax(resta)]    # 4: abajo-izquierda

    # Verificación: tienen que ser 4 puntos distintos
    if len(np.unique(ordenado, axis=0)) != 4:
        return None
    return ordenado

img = cv2.imread("d3prueba.jpg")
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# 1. Máscara (los mismos valores que te funcionaron)
mascara = cv2.inRange(hsv, (0, 0, 160), (179, 255, 255))

# 2. Limpiar
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel, iterations=2)
mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, kernel)

# 3. Contorno más grande, envuelto en su casco convexo
contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
mayor = max(contornos, key=cv2.contourArea)
casco = cv2.convexHull(mayor)

# 4. Simplificar a polígono
epsilon = 0.02 * cv2.arcLength(casco, True)
aprox = cv2.approxPolyDP(casco, epsilon, True)
print("Puntos:", len(aprox))

# 5. Si son 4, ordenarlas y numerarlas
vis = img.copy()
cv2.drawContours(vis, [casco], -1, (0, 255, 0), 2)
if len(aprox) == 4:
    esquinas = ordenar_esquinas(aprox)
    for i, (x, y) in enumerate(esquinas, start=1):
        cv2.circle(vis, (int(x), int(y)), 6, (0, 0, 255), -1)
        cv2.putText(vis, str(i), (int(x) + 8, int(y) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

cv2.imshow("mascara limpia", mascara)
cv2.imshow("resultado", vis)
cv2.waitKey(0)
cv2.destroyAllWindows()