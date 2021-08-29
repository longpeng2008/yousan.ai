#coding:utf8

# Copyright 2019 longpeng2008. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# If you find any problem,please contact us
#
#     longpeng2008to2012@gmail.com 
#
# or create issues
# =============================================================================
#coding:utf8
import os
import sys
import hashlib

# 获取文件的md5 
def get_md5(file):
    file = open(file,'rb')
    md5 = hashlib.md5(file.read())
    file.close()
    md5_values = md5.hexdigest()
    return md5_values


from collections import Counter
import numpy as np
import cv2
# 计算图像相似度
def compare_image(image1,image2,mode='same'):
    ## 比较是否完全相同
    if mode == 'same':
        assert(image1.shape == image2.shape)
        diff = (image1 == image2).astype(np.int)
        if cv2.countNonZero(diff) == image1.shape[0]*image1.shape[1]:
            return 1.0

    ## 比较是否基于阈值
    elif mode == 'abs':
        assert(image1.shape == image2.shape)
        diff = np.sum(np.abs((image1.astype(np.float) - image2.astype(np.float))))
        return diff / (image1.shape[0]*image1.shape[1])

    return 0
         
# 单文件夹去重
def remove_by_md5_singledir(file_dir): 
    file_list = os.listdir(file_dir)
    md5_list =[]
    print("去重前图像数量："+str(len(file_list)))
    for filepath in file_list:
        filemd5 = get_md5(os.path.join(file_dir,filepath))
        if filemd5 not in md5_list:
            md5_list.append(filemd5)
        else:
            os.remove(os.path.join(file_dir,filepath))

    print("去重后图像数量："+str(len(os.listdir(file_dir))))


# 单文件夹去重
def remove_by_pixel_singledir(file_dir,mode,th=5.0): 
    file_list = os.listdir(file_dir)
    print("去重前图像数量："+str(len(file_list)))
    for i in range(0,len(file_list)):
        if i < len(file_list)-1:
            imagei = cv2.imread(os.path.join(file_dir,file_list[i]),0)
            imagei = cv2.resize(imagei,(128,128),interpolation=cv2.INTER_NEAREST)
            print("testing image "+os.path.join(file_dir,file_list[i]))
            for j in range(i+1,len(file_list)):
                if j < len(file_list):
                    while j < len(file_list):
                        imagej = cv2.imread(os.path.join(file_dir,file_list[j]),0)
                        imagej = cv2.resize(imagej,(128,128),interpolation=cv2.INTER_NEAREST)
                        similarity = compare_image(imagei,imagej,mode=mode)
                        print("simi="+str(similarity))
                        if similarity >= 1.0 and mode == 'same':
                            os.remove(os.path.join(file_dir,file_list[j]))
                            print('删除'+os.path.join(file_dir,file_list[j]))
                            file_list.pop(j)
                        elif similarity < th and mode == 'abs':
                            os.remove(os.path.join(file_dir,file_list[j]))
                            print('删除'+os.path.join(file_dir,file_list[j]))
                            file_list.pop(j)
                        else:
                            break

    print("去重后图像数量："+str(len(os.listdir(file_dir))))

# 多文件夹去重
def remove_by_md5_multidir(file_list): 
    md5_list =[]
    print("去重前图像数量："+str(len(file_list)))
    for filepath in file_list:
        filemd5 = get_md5(filepath)
        file_id = filepath.split('/')[-1]
        file_dir = filepath[0:len(filepath)-len(file_id)]
        if filemd5 not in md5_list:
            md5_list.append(filemd5)
        else:
            os.remove(filepath)
    print("去重后图像数量："+str(len(md5_list)))

if __name__ == '__main__':
    file_dir = sys.argv[1]
    #remove_by_md5_singledir(file_dir)
    remove_by_pixel_singledir(file_dir,mode='abs',th=10)

    '''
    file_dir1 = sys.argv[1]
    file_list1 = os.listdir(file_dir1)
    file_list1 = [ os.path.join(file_dir1,x) for x in file_list1 ]
    file_dir2 = sys.argv[2]
    file_list2 = os.listdir(file_dir2)
    file_list2 = [ os.path.join(file_dir2,x) for x in file_list2 ]
    remove_by_md5_multidir(file_list1+file_list2)
    '''
