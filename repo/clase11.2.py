import cv2
import numpy as np

img = cv2.imread("Captura de pantalla 2026-10-26 141036.jpg")
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

cv2.namedWindow("mascara")
cv2.createTrackbar("H min", "mascara", 5, 179, lambda x: None)
cv2.createTrackbar("H max", "mascara", 20, 179, lambda x: None)
cv2.createTrackbar("S min", "mascara", 150, 255, lambda x: None)
cv2.createTrackbar("V min", "mascara", 120, 255, lambda x: None)

while True:
    h_min = cv2.getTrackbarPos("H min", "mascara")
    h_max = cv2.getTrackbarPos("H max", "mascara")
    s_min = cv2.getTrackbarPos("S min", "mascara")
    v_min = cv2.getTrackbarPos("V min", "mascara")

    mascara = cv2.inRange(hsv, (h_min, s_min, v_min), (h_max, 255, 255))

    cv2.imshow("original", img)
    cv2.imshow("mascara", mascara)
    if cv2.waitKey(30) == ord("q"):
        break

cv2.destroyAllWindows()
print("Valores:", h_min, h_max, s_min, v_min)