#coding:utf8

# Copyright 2020 longpeng2008. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# If you find any problem,please contact us
#
#     longpeng2008to2012@gmail.com 
#
# or create issues
# =============================================================================

import cv2
import numpy as np
from math import sqrt
import sys

# Read images : src image will be cloned into dst
im = cv2.imread(sys.argv[1])
obj = cv2.imread(sys.argv[2])
mask = 255 * np.ones(obj.shape, obj.dtype)
 
height,width,channels = im.shape
center = (width//2, height//2)
 
normal_clone = cv2.seamlessClone(obj, im, mask, center, cv2.NORMAL_CLONE)
mixed_clone = cv2.seamlessClone(obj, im, mask, center, cv2.MIXED_CLONE)
 
cv2.imwrite(sys.argv[3], mixed_clone)
cv2.imwrite(sys.argv[4], normal_clone)

