import cv2

img = cv2.imread('BanderaMexico.png', cv2.IMREAD_GRAYSCALE)

x, y, w, h = 500, 150, 150, 150
template = img[y:y+h, x:x+w]

cv2.imwrite('template.png', template)