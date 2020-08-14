# _*_ coding:utf8 _*_
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

