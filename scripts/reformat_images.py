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

def listfiles(rootDir,rename=False):
    list_dirs = os.walk(rootDir)
    num = 0
    for root, dirs, files in list_dirs:
        for d in dirs:
            print os.path.join(root,d)
        for f in files:
            fileid = f.split('.')[0] 
            filepath = os.path.join(root,f)
            try:
                src = cv2.imread(filepath,1)
                print "src=",filepath,src.shape
                os.remove(filepath) #删除原来图片
                if rename:
                    cv2.imwrite(os.path.join(root,str(num)+".jpg"),src) #写入新的图片，重新命名
                    num = num + 1
                else:
                    cv2.imwrite(os.path.join(root,fileid+".jpg"),src) #写入新的图片，名字不变
            except:
                os.remove(filepath) #去除损坏图片
                continue

listfiles(sys.argv[1],rename=True)

