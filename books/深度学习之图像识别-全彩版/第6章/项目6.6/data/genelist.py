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
def listfiles(rootDir,txtfile):
    ftxtfile = open(txtfile,'w')
    list_dirs = os.walk(rootDir)
    count = 0
    dircount = 0
    for root, dirs, files in list_dirs:
        for d in dirs:
            print os.path.join(root,d)
            dircount = dircount + 1
        for f in files:
            print os.path.join(root,f)
            ftxtfile.write(os.path.join(root,f)+'\n')
            count = count + 1
    print rootDir+"has"+str(count)+"files"

if __name__ == '__main__':
    listfiles('mask','train.txt')
