import cv2
import numpy as np
import matplotlib.pyplot as plt

# ---------------- Cargar imagen a color ----------------
img = cv2.imread('BanderaMexico.png')

# ---------- DIBUJAR SOBRE LA IMAGEN ----------

# Dibujar un rectángulo (ej. para marcar una región de interés)
cv2.rectangle(img, (50, 50), (200, 200), (0, 255, 0), 2)

# Dibujar un círculo
cv2.circle(img, (500, 285), 170, (255, 0, 0), 2)

# Dibujar una línea
cv2.line(img, (100, 300), (900, 400), (0, 0, 255), 2)

# Escribir texto
cv2.putText(img, 'Region de Interes', (50, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

# ---------- ROI (Región de Interés) ----------
# Definir coordenadas de la ROI (por ejemplo, el mismo rectángulo que dibujaste)
roi = img[50:200, 50:200]

# ---------- Mostrar con matplotlib ----------
plt.figure(figsize=(10, 5))

# Imagen con dibujos
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Imagen con Dibujos')
plt.axis('off')

# ROI
plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
plt.title('Región de Interés (ROI)')
plt.axis('off')

plt.tight_layout()
plt.show()