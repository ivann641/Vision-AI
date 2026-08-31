import cv2

# Tamaño del patrón: esquinas INTERNAS del tablero (9 de ancho, 6 de alto)
PATRON = (8, 8)

cap = cv2.VideoCapture(0)

while True:
    ok, frame = cap.read()
    if not ok:
        break

    # Paso 1: pasar a escala de grises (findChessboardCorners lo necesita así)
    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Paso 2: buscar las esquinas del tablero en la imagen gris
    # encontrado = True/False, esquinas = coordenadas de las 54 esquinas si las hay
    encontrado, esquinas = cv2.findChessboardCorners(gris, PATRON)

    # Paso 3: si las encontró, dibujarlas sobre el frame a color
    if encontrado:
        cv2.drawChessboardCorners(frame, PATRON, esquinas, encontrado)
        print("Tablero detectado")
    else:
        print("No se detecto el tablero")

    # Paso 4: mostrar el video (esto ya lo conocés de camara2.py)
    cv2.imshow("calibracion", frame)
    cv2.imshow("gris", gris)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()