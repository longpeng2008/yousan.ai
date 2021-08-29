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
import random

#file_in 输入txt文件
#file_out 输出txt文件

def shuffle(file_in,file_out):
    fin = open(file_in,'r')
    fout = open(file_out,'w')

    lines = fin.readlines()
    random.shuffle(lines)
    for line in lines:
        fout.write(line)

shuffle(sys.argv[1],sys.argv[2])
