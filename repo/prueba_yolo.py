from ultralytics import YOLO
import cv2

RUTA_FOTO = "dia2.jpg"

model = YOLO("best.pt")
img = cv2.imread(RUTA_FOTO)
if img is None:
    raise FileNotFoundError(f"No encontré {RUTA_FOTO} (revisá el nombre y la carpeta)")

res = model(img, conf=0.25)[0]
print("Cajas encontradas:", len(res.boxes))

for caja in res.boxes:
    x1, y1, x2, y2 = map(int, caja.xyxy[0].tolist())
    conf = float(caja.conf[0])
    print("Caja:", x1, y1, x2, y2, "confianza:", round(conf, 2))
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(img, f"{conf:.2f}", (x1, max(y1 - 8, 15)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

cv2.imshow("detecciones", img)
cv2.waitKey(0)
cv2.destroyAllWindows()