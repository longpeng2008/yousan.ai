import tensorflow as tf
from net import simpleconv3net
import sys
import numpy as np
import cv2
import os

testsize = 48
x = tf.placeholder(tf.float32, [1,testsize,testsize,3])
y = simpleconv3net(x)
y = tf.nn.softmax(y)

lines = open(sys.argv[2]).readlines()
count = 0
acc = 0
posacc = 0
negacc = 0
poscount = 0
negcount = 0

with tf.Session() as sess:  
    init = tf.global_variables_initializer()
    sess.run(init)  
    saver = tf.train.Saver()
    saver.restore(sess,sys.argv[1])
    
    #test one by one, you can change it into batch inputs
    for line in lines:
        imagename,label = line.strip().split(' ')
        #img = cv2.imread(imagename).astype(np.float32)
        #img = cv2.resize(img,(testsize,testsize),interpolation=cv2.INTER_NEAREST)
        
        img = tf.read_file(imagename)
        img = tf.image.decode_jpeg(img,channels = 3)
        img = tf.image.convert_image_dtype(img,dtype = tf.float32)
        img = tf.image.resize_images(img,(testsize,testsize),method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        img = tf.image.per_image_standardization(img)

        imgnumpy = img.eval()
        imgs = np.zeros([1,testsize,testsize,3],dtype=np.float32)
        imgs[0:1,] = imgnumpy

        result = sess.run(y, feed_dict={x:imgs})
        result = np.squeeze(result)
        if result[0] > result[1]:
            predict = 0
        else:
            predict = 1

        count = count + 1
        if str(predict) == '0':
            negcount = negcount + 1
            if str(label) == str(predict):
                negacc = negacc + 1
                acc = acc + 1
        else:
            poscount = poscount + 1
            if str(label) == str(predict):
                posacc = posacc + 1
                acc = acc + 1
    
        print result
print "acc = ",float(acc) / float(count)
print "poscount=",poscount
print "posacc = ",float(posacc) / float(poscount)
print "negcount=",negcount
print "negacc = ",float(negacc) / float(negcount)
