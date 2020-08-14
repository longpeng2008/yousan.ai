# _*_ coding:utf8 _*_
import os
import sys
import base64
import json
import math
import cv2
import random

def shuffle(file_in,file_out,num):
    ##num ,output num
    count = 0
    fin = open(file_in,'r')
    fout = open(file_out,'w')

    strings_in=[]
    strings_out=[]
    indexs_in = []
    while 1:
        line = fin.readline().strip()
        if line:
            count = count + 1
            strings_in.append(line)
        else:
            break
    
    print "-----there is-----"+str(count)+'-----object in filein'
    if num > count:
        print "-----the select num-----"+str(num)+'-----is bigger than files in file_in'
        return
    count_select = 0
    while 1:
        if count_select >=num:
            break
        index = int(random.uniform(0,count))
        if not index in indexs_in:
            indexs_in.append(index)
            fout.write(strings_in[index])
            fout.write('\n')
            count_select = count_select + 1
        else:
            continue

shuffle(sys.argv[1],sys.argv[2],int(sys.argv[3]))
