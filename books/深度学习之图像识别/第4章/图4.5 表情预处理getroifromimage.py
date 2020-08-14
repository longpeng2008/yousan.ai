#coding:utf8
import cv2
import dlib
import numpy as np
import sys
import os
sys.path.insert(0, '/home/longpeng/opts/1_Caffe_Long/python/')
import caffe

PREDICTOR_PATH = "/home/longpeng/project/3DFace/3dmm_cnn/dlib_model/shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(PREDICTOR_PATH)
cascade_path='/home/longpeng/opts/opencv3.2/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml'
cascade = cv2.CascadeClassifier(cascade_path)


def get_landmarks(im):
    rects = cascade.detectMultiScale(im, 1.3,5)
    x,y,w,h =rects[0]
    rect=dlib.rectangle(x,y,x+w,y+h)
    return np.matrix([[p.x, p.y] for p in predictor(im, rect).parts()])

def annotate_landmarks(im, landmarks):
    im = im.copy()
    for idx, point in enumerate(landmarks):
        pos = (point[0, 0], point[0, 1])
        cv2.putText(im, str(idx), pos,
                    fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                    fontScale=0.4,
                    color=(0, 0, 255))
        cv2.circle(im, pos, 5, color=(0, 255, 255))
    return im

im=cv2.imread(sys.argv[1],1)
cv2.namedWindow('Result',0)
cv2.imshow('Result',annotate_landmarks(im,get_landmarks(im)))

print "image shape=",im.shape
landmarks = get_landmarks(im)
print "landmarks",landmarks.shape

xmin = 10000
xmax = 0
ymin = 10000
ymax = 0

for i in range(48,67):
    x = landmarks[i,0]
    y = landmarks[i,1]
    if x < xmin:
        xmin = x
    if x > xmax:
        xmax = x
    if y < ymin:
        ymin = y
    if y > ymax:
        ymax = y

print "xmin=",xmin
print "xmax=",xmax
print "ymin=",ymin
print "ymax=",ymax

roiwidth = xmax - xmin
roiheight = ymax - ymin

roi = im[ymin:ymax,xmin:xmax,0:3]

if roiwidth > roiheight:
    dstlen = 1.5*roiwidth
else:
    dstlen = 1.5*roiheight

diff_xlen = dstlen - roiwidth
diff_ylen = dstlen - roiheight

newx = xmin
newy = ymin

imagerows,imagecols,channel = im.shape
if newx >= diff_xlen/2 and newx + roiwidth + diff_xlen/2 < imagecols:
    newx  = newx - diff_xlen/2;
elif newx < diff_xlen/2:
    newx = 0;
else:
    newx =  imagecols - dstlen;

if newy >= diff_ylen/2 and newy + roiheight + diff_ylen/2 < imagerows:
    newy  = newy - diff_ylen/2;
elif newy < diff_ylen/2:
    newy = 0;
else:
    newy =  imagerows - dstlen;

roi = im[int(newy):int(newy+dstlen),int(newx):int(newx+dstlen),0:3]
