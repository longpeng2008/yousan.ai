#coding:utf8

# Copyright 2020 longpeng2008. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# If you find any problem,please contact us
#
#     longpeng2008to2012@gmail.com 
#
# or create issues
# =============================================================================

import math
import sys
import os
import numpy as np
import cv2
import time

def integral(img):
    integ_graph = np.zeros((img.shape[0],img.shape[1]),dtype = np.float32)
    for x in range(img.shape[0]):
        sum_clo = 0
        for y in range(img.shape[1]):
            sum_clo = sum_clo + img[x][y]
            integ_graph[x][y] = integ_graph[x-1][y] + sum_clo;
    return integ_graph

def search_rectangle(saliency_map,infor_ratio_th,is_keep_aspect):
    srcimage = np.copy(saliency_map)
    saliency_map = saliency_map.astype(np.float32) / np.sum(saliency_map)
    integral_map = integral(saliency_map)
    
    xspeed_add = 0
    yspeed_add = 0
    xspeed_sub = 0
    yspeed_sub = 0

    if len(saliency_map.shape) !=2:
        print "saliency_map must be single channel"
        return 
    
    height,width = saliency_map.shape
    x_start = 0
    y_start = 0
    x_end = width-1
    y_end = height-1
    
    total_infor = integral_map[height-1,width-1]
    cur_infor = 0
    flag = 0
    

    step = 5
    while 1:
        xspeed_add = integral_map[y_start,x_start] + integral_map[y_end,x_start+step] - integral_map[y_end,x_start] - integral_map[y_start,x_start+step]
        yspeed_add = integral_map[y_start,x_start] + integral_map[y_start+step,x_end] - integral_map[y_start+step,x_start] - integral_map[y_start,x_end]

        xspeed_sub = integral_map[y_end,x_end] + integral_map[y_start,x_end-step] - integral_map[y_end,x_end-step] - integral_map[y_start,x_end]
        yspeed_sub = integral_map[y_end,x_end] + integral_map[y_end-step,x_start] - integral_map[y_end,x_start] - integral_map[y_end-step,x_end]
        
        tmp = [xspeed_add,yspeed_add,xspeed_sub,yspeed_sub]
        index = np.argmin(tmp)

        if index == 0: ##add x index
            x_start = x_start + step
        elif index == 1:
            y_start = y_start + step
        elif index == 2:
            x_end = x_end - step
        else:
            y_end = y_end - step

        cur_infor = integral_map[y_end,x_end] - integral_map[y_start,x_start]
        
        if cur_infor < infor_ratio_th * total_infor:
            break

    return [x_start,y_start,x_end,y_end]

if __name__ == "__main__":
    saliency_map = cv2.imread(sys.argv[1],0)

    imgdir = '../../../datasets/cocos/label/'
    images = os.listdir(imgdir)
    
    th = 0.95

    for image in images:
        saliency_map = cv2.imread(os.path.join(imgdir,image),0)

        start = cv2.getTickCount()
        rect = search_rectangle(saliency_map,0.95,True)
        end = cv2.getTickCount()
        print "used time=",(end-start)*1000/cv2.getTickFrequency(),"ms"

    print "result=",rect



