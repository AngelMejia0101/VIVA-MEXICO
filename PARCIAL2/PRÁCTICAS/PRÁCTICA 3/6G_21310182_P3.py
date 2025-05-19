import cv2
import numpy as np
import matplotlib.pyplot as plt

# ---------------- Cargar imagen a color ----------------
img_color = cv2.imread('BanderaMexico.png')

# Convertir la imagen a escala de grises para análisis
img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

# ---------------- Imagen modificada ----------------
img_modificada = cv2.add(img_gray, 50)

# ---------------- Histograma de imagen modificada ----------------
hist_mod = cv2.calcHist([img_modificada], [0], None, [256], [0, 256])

# ---------------- Ecualización ----------------
img_eq = cv2.equalizeHist(img_modificada)
hist_eq = cv2.calcHist([img_eq], [0], None, [256], [0, 256])

# ---------------- Función para cerrar con tecla ----------------
def cerrar_ventana(event):
    if event.key == '0':
        plt.close()

# ---------------- Visualización ----------------
fig = plt.figure(figsize=(12, 6))
fig.canvas.mpl_connect('key_press_event', cerrar_ventana)

plt.subplot(2, 2, 1)
plt.imshow(cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB))
plt.title('Imagen Original (Color)')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.plot(hist_mod, color='gray')
plt.title('Histograma - Imagen Modificada (Suma)')
plt.xlabel('Intensidad')
plt.ylabel('Frecuencia')

plt.subplot(2, 2, 3)
plt.imshow(img_eq, cmap='gray')
plt.title('Imagen Ecualizada')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.plot(hist_eq, color='gray')
plt.title('Histograma - Imagen Ecualizada')
plt.xlabel('Intensidad')
plt.ylabel('Frecuencia')

plt.tight_layout()
plt.show()
