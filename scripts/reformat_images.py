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
import numpy as np

def listfiles(rootDir):
    list_dirs = os.walk(rootDir)
    for root, dirs, files in list_dirs:
        for d in dirs:
            print os.path.join(root,d)
        for f in files:
                
            fileid = f.split('.')[0] 

            filepath = os.path.join(root,f)
            try:
                src = cv2.imread(filepath,1)
                print "src=",filepath,src.shape
                os.remove(filepath)
                cv2.imwrite(os.path.join(root,fileid+".jpg"),src)
            except:
                os.remove(filepath)
                continue

listfiles(sys.argv[1])

