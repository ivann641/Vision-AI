import cv2

captura = cv2.VideoCapture(0)

while True:
    ret, frame = captura.read()
    if not ret:
        break

    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    resultado = cv2.adaptiveThreshold(gris, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, -20)

    cv2.imshow("Original", frame)
    cv2.imshow("Adaptive Threshold", resultado)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

captura.release()
cv2.destroyAllWindows()