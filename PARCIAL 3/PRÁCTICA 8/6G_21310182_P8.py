
import cv2  
import numpy as np  
import matplotlib.pyplot as plt  

img = cv2.imread('BanderaMexico.png', cv2.IMREAD_GRAYSCALE)

laplaciano = cv2.Laplacian(img, cv2.CV_64F)
laplaciano = cv2.convertScaleAbs(laplaciano)

sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
sobelx = cv2.convertScaleAbs(sobelx)

sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
sobely = cv2.convertScaleAbs(sobely)

canny = cv2.Canny(img, 100, 200)
plt.figure(figsize=(10, 6))  

plt.subplot(2, 3, 1)
plt.imshow(img, cmap='gray')
plt.title('Imagen Original')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(laplaciano, cmap='gray')
plt.title('Laplaciano')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(sobelx, cmap='gray')
plt.title('Sobel X')
plt.axis('off')

plt.subplot(2, 3, 4)
plt.imshow(sobely, cmap='gray')
plt.title('Sobel Y')
plt.axis('off')

plt.subplot(2, 3, 5)
plt.imshow(canny, cmap='gray')
plt.title('Canny')
plt.axis('off')

plt.tight_layout()
plt.show()