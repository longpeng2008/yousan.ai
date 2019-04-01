#coding:utf8

# Copyright 2019 longpeng2008. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# If you find any problem,please contact us
#
#     longpeng2008to2012@gmail.com 
#
# or create issues
# =============================================================================


import os
import sys
import cv2
for image in os.listdir(sys.argv[1]):
    img = cv2.imread(os.path.join(sys.argv[1],image))
    print image," shape is ",img.shape

