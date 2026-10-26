import cv2
import numpy as np

RUTA_FOTO = "dia2.jpg"              # <- la misma foto
RECORTE = (15, 405, 1060, 1280)     # (y1, y2, x1, x2) del cartel violeta

# Parámetros de la máscara: para experimentar, cambiá de a uno
H_MIN, H_MAX = 0, 179
S_MIN = 100
V_MIN = 60


def ordenar_esquinas(pts):
    pts = np.array(pts, dtype=np.float32).reshape(4, 2)
    suma = pts.sum(axis=1)
    resta = np.diff(pts, axis=1).ravel()
    ordenado = np.zeros((4, 2), dtype=np.float32)
    ordenado[0] = pts[np.argmin(suma)]
    ordenado[1] = pts[np.argmin(resta)]
    ordenado[2] = pts[np.argmax(suma)]
    ordenado[3] = pts[np.argmax(resta)]
    if len(np.unique(ordenado, axis=0)) != 4:
        return None
    return ordenado


def interseccion(a, b, c, d):
    """Punto donde se cruzan la recta a-b y la recta c-d (None si son paralelas)."""
    r1 = np.cross([a[0], a[1], 1.0], [b[0], b[1], 1.0])   # recta que pasa por a y b
    r2 = np.cross([c[0], c[1], 1.0], [d[0], d[1], 1.0])   # recta que pasa por c y d
    p = np.cross(r1, r2)                                   # cruce de las dos rectas
    if abs(p[2]) < 1e-6:
        return None
    return np.array([p[0] / p[2], p[1] / p[2]], dtype=np.float32)


def esquinas_desde_cinco(pts5, margen=0.3):
    pts = np.array(pts5, dtype=np.float32).reshape(5, 2)

    # 1. La diagonal es el lado cuyo punto medio tiene x+y más grande
    medios = (pts + np.roll(pts, -1, axis=0)) / 2
    i = int(np.argmax(medios.sum(axis=1)))       # la diagonal une pts[i] con pts[i+1]

    # 2. Prolongar el lado anterior y el siguiente a la diagonal
    cruce = interseccion(pts[(i - 1) % 5], pts[i], pts[(i + 1) % 5], pts[(i + 2) % 5])
    if cruce is None:
        return None

    # 3. Si el cruce cae lejísimo del cartel, no lo creemos
    xmin, ymin = pts.min(axis=0)
    xmax, ymax = pts.max(axis=0)
    mx, my = margen * (xmax - xmin), margen * (ymax - ymin)
    if not (xmin - mx <= cruce[0] <= xmax + mx and ymin - my <= cruce[1] <= ymax + my):
        return None

    # 4. Sacar los 2 puntos de la diagonal y poner el cruce
    return np.array([pts[(i + 2) % 5], pts[(i + 3) % 5], pts[(i + 4) % 5], cruce],
                    dtype=np.float32)


foto = cv2.imread(RUTA_FOTO)
if foto is None:
    raise FileNotFoundError(f"No encontré {RUTA_FOTO}")

y1, y2, x1, x2 = RECORTE
img = foto[y1:y2, x1:x2]

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
mascara = cv2.inRange(hsv, (H_MIN, S_MIN, V_MIN), (H_MAX, 255, 255))

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel, iterations=2)
mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, kernel)

contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if not contornos:
    raise RuntimeError("No se encontró ningún contorno")
mayor = max(contornos, key=cv2.contourArea)
casco = cv2.convexHull(mayor)

aprox = cv2.approxPolyDP(casco, 0.02 * cv2.arcLength(casco, True), True)
print("Puntos:", len(aprox))

vis = img.copy()
cv2.drawContours(vis, [casco], -1, (0, 255, 0), 2)

if len(aprox) == 4:
    esquinas = ordenar_esquinas(aprox)
elif len(aprox) == 5:
    cuatro = esquinas_desde_cinco(aprox)
    esquinas = ordenar_esquinas(cuatro) if cuatro is not None else None
else:
    esquinas = None
    print("Ni 4 ni 5 puntos, no se ordena")

if esquinas is None:
    print("No se pudieron obtener las 4 esquinas")
else:
    for i, (x, y) in enumerate(esquinas, start=1):
        cv2.circle(vis, (int(x), int(y)), 6, (0, 0, 255), -1)
        cv2.putText(vis, str(i), (int(x) + 8, int(y) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

cv2.imshow("recorte", img)
cv2.imshow("mascara limpia", mascara)
cv2.imshow("resultado", vis)
cv2.waitKey(0)
cv2.destroyAllWindows()