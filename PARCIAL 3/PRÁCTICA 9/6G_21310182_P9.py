import cv2
import numpy as np

img = cv2.imread('BanderaMexico.png', cv2.IMREAD_GRAYSCALE)
template = cv2.imread('template.png', cv2.IMREAD_GRAYSCALE)

x, y, w, h = 500, 150, 150, 150  
template = img[y:y+h, x:x+w]

result = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
threshold = 0.8  
loc = np.where(result >= threshold)

img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
detections = 0
for pt in zip(*loc[::-1]):  # (x, y)
    cv2.rectangle(img_color, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)
    detections += 1

print(f"Coincidencias detectadas: {detections}")

cv2.imshow('Detecciones', img_color)
cv2.waitKey(0)
cv2.destroyAllWindows()
