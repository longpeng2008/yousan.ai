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
import glob
import numpy as np
import cv2
 
class Dataset:
    def __init__(self, rootpath, imgwidth, imgheight, trainratio=0.9):
        self.rootpath = rootpath
        list_dirs = os.walk(self.rootpath)
        count = 0
        numofclasses = 0
	self.subdirs = []
        for root, dirs, files in list_dirs:
            for d in dirs:
                self.subdirs.append(os.path.join(root,d))
		#print os.path.join(root,d)
                numofclasses = numofclasses + 1
	#print "numofclasses",numofclasses	
        label = 0
	self.imagedatas = []
	self.labeldatas = []
	for subdir in self.subdirs:
	    images = glob.iglob(os.path.join(subdir,'*.jpg'))
	    for image in images:  
                #print image
                #h,w,c
                imagedata = cv2.imread(image,1)
                imagedata = cv2.resize(imagedata,(imgwidth,imgheight)) 
                imagedata = imagedata.astype(np.float) / 255.0
                imagedata = imagedata - [0.5,0.5,0.5]
                imagedata = imagedata.transpose((2,0,1))

		self.imagedatas.append(imagedata)
		self.labeldatas.append(label)

                #self.imagedatas = np.concatenate((self.imagedatas, imagedata), axis = 0)
                #self.labeldatas = np.concatenate((self.labeldatas, label), axis = 0)
	    label = label + 1

	print "length of images=",len(self.imagedatas)
	print "length of labels=",len(self.labeldatas)

	self.imagedatas = np.array(self.imagedatas).astype(np.float32)
	self.labeldatas = np.array(self.labeldatas).astype(np.int32)
        

        indices = np.arange(len(self.imagedatas))
        np.random.shuffle(indices)
        splitindex = int(trainratio*self.imagedatas.shape[0])
        
        self.imagetraindatas = self.imagedatas[0:splitindex].copy()
        self.labeltraindatas = self.labeldatas[0:splitindex].copy()
        
        self.imagevaldatas = self.imagedatas[splitindex:].copy()
        self.labelvaldatas = self.labeldatas[splitindex:].copy()
        
    def getlen(self):
        return len(self.imagedatas)
    def gettrainlen(self):
        return len(self.imagetraindatas)
    def getvallen(self):
        return len(self.imagevaldatas)


    def iterate_minibatches(self, inputs, targets, batchsize, shuffle=False):
        assert len(inputs) == len(targets)
        if shuffle:
            indices = np.arange(len(inputs))
            print "indices type=",type(indices)
            np.random.shuffle(indices)
        for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batchsize]
            else:
                excerpt = slice(start_idx, start_idx + batchsize)
            yield inputs[excerpt], targets[excerpt]
   
if __name__ == "__main__":
 
    mydataset = Dataset(sys.argv[1],48,48)
    batches =  mydataset.iterate_minibatches(mydataset.imagedatas,mydataset.labeldatas,16,True)

    index = 0
    for batch in batches:
        inputs, targets = batch
        print len(inputs),index,targets
        index = index + 1
