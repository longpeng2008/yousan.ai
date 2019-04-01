#_*_ coding:utf8
import sys
sys.path.insert(0, '../caffe/python/')
import caffe
import os,shutil
import numpy as np
from PIL import Image as PILImage
from PIL import ImageMath
import matplotlib.pyplot as plt
import time
import cv2

debug=True
import argparse
def parse_args():
   parser = argparse.ArgumentParser(description='test resnet model for portrait segmentation')
   parser.add_argument('--model', dest='model_proto', help='the model', default='test.prototxt', type=str)
   parser.add_argument('--weights', dest='model_weight', help='the weights', default='./test.caffemodel', type=str)
   parser.add_argument('--testsize', dest='testsize', help='inference size', default=60,type=int)
   parser.add_argument('--src', dest='img_folder', help='the src image folder', type=str, default='./')
   parser.add_argument('--gt', dest='gt', help='the gt', type=int, default=0)
   args = parser.parse_args()
   return args

def start_test(model_proto,model_weight,img_folder,testsize):
   caffe.set_device(0)
   #caffe.set_mode_cpu()
   net = caffe.Net(model_proto, model_weight, caffe.TEST)
   imgs = os.listdir(img_folder)
   
   pos = 0
   neg = 0

   for imgname in imgs:
      imgtype = imgname.split('.')[-1]
      imgid = imgname.split('.')[0]
      if imgtype != 'png' and imgtype != 'jpg' and imgtype != 'JPG' and imgtype != 'jpeg' and imgtype != 'tif' and imgtype != 'bmp':
          print imgtype,"error"
          continue
      imgpath = os.path.join(img_folder,imgname)

      img = cv2.imread(imgpath)
      if img is None:
          print "---------img is empty---------",imgpath
          continue
      
      img = cv2.resize(img,(testsize,testsize))

      transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
      transformer.set_mean('data', np.array([104.008,116.669,122.675]))
      transformer.set_transpose('data', (2,0,1))
              
      out = net.forward_all(data=np.asarray([transformer.preprocess('data', img)]))
         
      result = out['prob'][0]
      print "---------result prob---------",result,"-------result size--------",result.shape
      probneutral = result[0]
      print "prob neutral",probneutral 
     
      probsmile = result[1]
      print "prob smile",probsmile
      problabel = -1
      probstr = 'none'
      if probneutral > probsmile:
          probstr = "neutral:"+str(probneutral)
          pos = pos + 1
      else:
          probstr = "smile:"+str(probsmile)
          neg = neg + 1
      
      if debug:
         showimg = cv2.resize(img,(256,256))
         cv2.putText(showimg,probstr,(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),1)
         cv2.imshow("test",showimg)
         k = cv2.waitKey(0)
         if k == ord('q'):
             break
   
   print "pos=",pos 
   print "neg=",neg 

if __name__ == '__main__':
    args = parse_args()
    start_test(args.model_proto,args.model_weight,args.img_folder,args.testsize)
