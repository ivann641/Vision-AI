import cv2

captura = cv2.VideoCapture(0)

while True:
    ret, frame = captura.read()
    if not ret:
        break

    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    resultado = cv2.adaptiveThreshold(gris, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 5)

    contornos, jerarquia = cv2.findContours(resultado, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contorno in contornos:
        perimetro = cv2.arcLength(contorno, True)
        epsilon = 0.02 * perimetro
        aproximacion = cv2.approxPolyDP(contorno, epsilon, True)

        if len(aproximacion) == 4:
            area = cv2.contourArea(contorno)
            if area > 200:
                if cv2.isContourConvex(aproximacion):
                    x, y, ancho, alto = cv2.boundingRect(aproximacion)
                    aspecto = ancho / float(alto)
                    if 0.7 < aspecto < 1.3:
                        cv2.drawContours(frame, [aproximacion], -1, (0, 255, 0), 3)

    cv2.imshow("Original", frame)
    cv2.imshow("Adaptive Threshold", resultado)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

captura.release()
cv2.destroyAllWindows()