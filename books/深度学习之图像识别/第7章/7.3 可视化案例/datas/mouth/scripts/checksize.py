import os
import sys
import cv2
for image in os.listdir(sys.argv[1]):
    img = cv2.imread(os.path.join(sys.argv[1],image))
    print image," shape is ",img.shape

