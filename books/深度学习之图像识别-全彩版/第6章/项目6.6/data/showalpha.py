import os
import sys
import cv2
img = cv2.imread(sys.argv[1],cv2.IMREAD_UNCHANGED)
h,w,c = img.shape
print('channels='+str(c))

alpha = img[:,:,c-1]
cv2.imshow('alpha',alpha)
cv2.imwrite('alpha.png',alpha)
cv2.waitKey(0)
