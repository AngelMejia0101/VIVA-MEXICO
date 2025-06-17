import cv2  
import numpy as np  

img_color = cv2.imread('BanderaMexico.png')  
img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)  

x, y, w, h = 500, 150, 150, 150

roi_color = img_color[y:y+h, x:x+w]  
roi_gray = img_gray[y:y+h, x:x+w]    

roi_gray = np.float32(roi_gray)

dst = cv2.cornerHarris(roi_gray, blockSize=2, ksize=3, k=0.04)
# blockSize: tamaño del vecindario considerado para detección
# ksize: tamaño del kernel de Sobel
# k: parámetro libre entre 0.04 y 0.06

dst = cv2.dilate(dst, None)


roi_color[dst > 0.01 * dst.max()] = [255, 0, 255]

cv2.imshow('ROI con esquinas detectadas', roi_color)
cv2.waitKey(0)  
cv2.destroyAllWindows()  