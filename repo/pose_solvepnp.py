import sys
from pathlib import Path
import numpy as np
import cv2
 
# ---------- 1. Medidas del cartel (en metros, medidas con cinta) ----------
ANCHO = 0.80   # M19. Cambialo por el cartel que uses en la prueba
ALTO = 0.95
 
def puntos_del_cartel(ancho, alto):
    """
    Esquinas del cartel en el mundo real (metros), con el origen en su centro
    y Z = 0 (el cartel es plano). Mismo orden que las esquinas de la imagen.
    """
    return np.array([
        [-ancho / 2, -alto / 2, 0],   # 1 arriba-izq
        [ ancho / 2, -alto / 2, 0],   # 2 arriba-der
        [ ancho / 2,  alto / 2, 0],   # 3 abajo-der
        [-ancho / 2,  alto / 2, 0],   # 4 abajo-izq
    ], dtype=np.float64)
 
 
puntos_pared = puntos_del_cartel(ANCHO, ALTO)
 
# ---------- 2. Calibracion ----------
# Resolucion (ancho, alto) con la que se calibro. Las fotos tienen que ser asi.
TAMANO_CALIBRACION = (1280, 720)
 
 
def cargar_calibracion(ruta="calibracion.npz"):
    """Devuelve la matriz de la camara (K) y los coeficientes de distorsion."""
    d = np.load(ruta)
    return d["matriz_intrinseca"], d["coef_distorsion"]
 
 
# ---------- 3. La cuenta: de 4 esquinas a distancia y angulo ----------
def calcular_pose(puntos_foto, K, distorsion, ancho=ANCHO, alto=ALTO, detalle=False):
    """
    puntos_foto: 4 esquinas en pixeles, en el orden arriba-izq, arriba-der,
                 abajo-der, abajo-izq.
    ancho, alto: medidas reales del cartel en metros.
    Devuelve (distancia_m, angulo_deg, error_reproyeccion_px).
    Con detalle=True devuelve ademas las 4 esquinas reproyectadas (donde el modelo
    cree que deberian estar), para compararlas con los clicks.
    """
    puntos_foto = np.asarray(puntos_foto, dtype=np.float64).reshape(4, 2)
    puntos_pared = puntos_del_cartel(ancho, alto)
 
    # solvePnP con el metodo IPPE, pensado para objetos planos.
    # La distorsion se pasa aca: NO hay que hacer undistort ademas.
    ok, rvec, tvec = cv2.solvePnP(puntos_pared, puntos_foto, K, distorsion,
                                  flags=cv2.SOLVEPNP_IPPE)
    if not ok:
        raise RuntimeError("solvePnP no pudo resolver la pose")
 
    # Posicion de la camara vista desde el cartel (origen en el centro del cartel)
    R, _ = cv2.Rodrigues(rvec)
    camara = (-R.T @ tvec).ravel()
    cx, cy, cz = camara
 
    distancia = float(np.sqrt(cx**2 + cz**2))             # en el piso, sin la altura
    angulo = float(np.degrees(np.arctan2(cx, -cz)))       # 0 = de frente; el signo dice de que lado
 
    # Chequeo: si la pose es buena, al proyectar las esquinas reales
    # tienen que caer cerca de donde se marcaron
    reproyectados, _ = cv2.projectPoints(puntos_pared, rvec, tvec, K, distorsion)
    error = float(np.linalg.norm(reproyectados.reshape(4, 2) - puntos_foto, axis=1).mean())
 
    if detalle:
        return distancia, angulo, error, reproyectados.reshape(4, 2)
    return distancia, angulo, error
 
 
def dibujar_resultado(img, clicks, repro):
    """Dibuja los clicks (rojo) y donde el modelo cree que deberian estar (verde)."""
    vista = img.copy()
    for i in range(4):
        c = (int(round(clicks[i][0])), int(round(clicks[i][1])))
        r = (int(round(repro[i][0])), int(round(repro[i][1])))
        cv2.line(vista, c, r, (255, 255, 0), 1)
        cv2.circle(vista, c, 6, (0, 0, 255), -1)
        cv2.circle(vista, r, 9, (0, 255, 0), 2)
        cv2.putText(vista, str(i + 1), (c[0] + 10, c[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    return vista
 
 
# ---------- 4. Programa principal: 4 clicks sobre una foto ----------
def main():
    if len(sys.argv) < 2:
        sys.exit("Uso: python pose_solvepnp.py foto.jpg [calibracion.npz]")
 
    ruta_foto = sys.argv[1]
    ruta_calibracion = sys.argv[2] if len(sys.argv) > 2 else "calibracion.npz"
 
    K, distorsion = cargar_calibracion(ruta_calibracion)
 
    img = cv2.imread(ruta_foto)
    if img is None:
        sys.exit(f"No pude abrir la foto: {ruta_foto}")
 
    # K solo sirve para la resolucion con la que se calibro
    alto, ancho = img.shape[:2]
    if (ancho, alto) != TAMANO_CALIBRACION:
        sys.exit(f"La foto es de {ancho}x{alto} y la calibracion es de "
                 f"{TAMANO_CALIBRACION[0]}x{TAMANO_CALIBRACION[1]}. "
                 f"Con otra resolucion la matriz K no corresponde.")
 
    clicks = []
 
    def al_clickear(evento, x, y, flags, param):
        if evento == cv2.EVENT_LBUTTONDOWN and len(clicks) < 4:
            clicks.append([x, y])
 
    cv2.namedWindow("foto")
    cv2.setMouseCallback("foto", al_clickear)
 
    while True:
        vista = img.copy()
        for i, (x, y) in enumerate(clicks):
            cv2.circle(vista, (x, y), 5, (0, 0, 255), -1)
            cv2.putText(vista, str(i + 1), (x + 8, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imshow("foto", vista)
        tecla = cv2.waitKey(20) & 0xFF
        if tecla == ord("r"):
            clicks.clear()
        if tecla == ord("q") or len(clicks) == 4:
            break
 
    cv2.waitKey(500)
    cv2.destroyAllWindows()
    if len(clicks) < 4:
        sys.exit("Faltan clicks.")
 
    distancia, angulo, error, repro = calcular_pose(clicks, K, distorsion, detalle=True)
 
    print(f"Cartel usado: {ANCHO} x {ALTO} m")
    print(f"Distancia: {distancia:.2f} m")
    print(f"Angulo:    {angulo:+.1f} grados")
    print(f"Error de reproyeccion: {error:.1f} px  ({'OK' if error <= 3 else 'REVISAR clicks'})")
 
    # Error de cada esquina: la que tenga mas error es la sospechosa
    errores = np.linalg.norm(repro - np.array(clicks, dtype=np.float64), axis=1)
    for i, e in enumerate(errores, start=1):
        print(f"   esquina {i}: {e:.1f} px")
 
    # Foto de control: rojo = tus clicks, verde = donde el modelo cree que deberian estar
    vista = dibujar_resultado(img, clicks, repro)
    salida = f"resultado_{Path(ruta_foto).stem}.jpg"
    cv2.imwrite(salida, vista)
    print(f"Guardada la foto de control: {salida}")
    cv2.imshow("rojo = tus clicks | verde = donde deberia estar (cualquier tecla para cerrar)", vista)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
 
 
if __name__ == "__main__":
    main()