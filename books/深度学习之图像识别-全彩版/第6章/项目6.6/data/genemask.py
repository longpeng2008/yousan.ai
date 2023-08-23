import cv2
import os
images = os.listdir('mask')
for image in images:
    img = cv2.imread('mask/'+image)
    img = (img > 0)*255
    cv2.imwrite('mask/'+image,img)
