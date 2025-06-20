import cv2  

img1 = cv2.imread('template.png', 0)  
img2 = cv2.imread('BanderaMexico.png', 0)     

orb = cv2.ORB_create()

kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

matches = bf.match(des1, des2)

matches = sorted(matches, key=lambda x: x.distance)

resultado = cv2.drawMatches(img1, kp1, img2, kp2, matches[:20], None, flags=2)

cv2.imshow('Similitudes con ORB', resultado)
cv2.waitKey(0)  
cv2.destroyAllWindows()  