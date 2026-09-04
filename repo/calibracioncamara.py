import cv2
import os

PATRON = (9, 6)
MINIMO_NECESARIO = 20
CARPETA_CAPTURAS = "capturas"

# Crea la carpeta si no existe (evita el error de "no such directory")
os.makedirs(CARPETA_CAPTURAS, exist_ok=True)

cap = cv2.VideoCapture(0)

puntos_guardados = []

while True:
    ok, frame = cap.read()
    if not ok:
        break

    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    encontrado, esquinas = cv2.findChessboardCorners(gris, PATRON)

    if encontrado:
        cv2.drawChessboardCorners(frame, PATRON, esquinas, encontrado)

    texto = f"Guardadas: {len(puntos_guardados)}/{MINIMO_NECESARIO}"
    cv2.putText(frame, texto, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("calibracion", frame)
    cv2.imshow("gris", gris)

    tecla = cv2.waitKey(1)

    if tecla == ord('s') and encontrado:
        puntos_guardados.append(esquinas)

        # Guardamos la imagen a color en disco, con nombre numerado
        numero = len(puntos_guardados)
        nombre_archivo = f"{CARPETA_CAPTURAS}/captura_{numero:02d}.jpg"
        cv2.imwrite(nombre_archivo, frame)

        print(f"Guardada #{numero} -> {nombre_archivo}")

    if tecla == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print(f"Total de detecciones guardadas: {len(puntos_guardados)}")