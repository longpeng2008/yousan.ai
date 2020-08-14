#coding:utf8

# Copyright 2019 longpeng2008. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# If you find any problem,please contact us
#
#     longpeng2008to2012@gmail.com 
#
# or create issues
# =============================================================================
import sys
import os
f=open(sys.argv[1],'r')
g=open(sys.argv[2],'w')
lines = f.readlines()
count = 0
for line in lines:
    src,label = line.strip().split(' ')
    g.write(str(count)+'\t'+str(label)+'\t'+src+'\n')
    count = count + 1

    
