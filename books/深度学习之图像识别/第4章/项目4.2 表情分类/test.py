#_*_ coding:utf8
import sys
sys.path.insert(0, '/home/longpeng/opts/caffe/python/')
import caffe
import os,shutil
import numpy as np
from PIL import Image as PILImage
from PIL import ImageMath
import matplotlib.pyplot as plt
import time
import cv2
pallete = [0,0,0,
        255, 255, 255]

debug=True
import argparse
def parse_args():
   parser = argparse.ArgumentParser(description='test resnet model for portrait segmentation')
   parser.add_argument('--model', dest='model_proto', help='the model', default='train_val_res18_matting_prob.prototxt', type=str)
   parser.add_argument('--weights', dest='model_weight', help='the weights', default='./res18_iter_25000.caffemodel', type=str)
   parser.add_argument('--showmodel', dest='showmodel', help='if show model', default=1,type=int)
   parser.add_argument('--modeltype', dest='modeltype', help='res18-origin or res18-momo', default='res18-momo',type=str)
   parser.add_argument('--testsize', dest='testsize', help='inference size', default=0,type=int)
   parser.add_argument('--src', dest='imgtxt', help='the src', type=str, default='all.txt')
   parser.add_argument('--dst', dest='out_path', help='the dst', type=str, default='./test_results/')
   parser.add_argument('--gt', dest='gt', help='the gt', type=int, default=0)
   parser.add_argument('--enable_choose', dest='enable_choose', help='enable choose', type=int, default=False)
   parser.add_argument('--enable_crop', dest='enable_crop', help='use crop or resize when test model', type=int, default=1)
   args = parser.parse_args()
   return args

def start_test(model_proto,model_weight,imgtxt,testsize,enable_crop):
   caffe.set_device(0)
   #caffe.set_mode_cpu()
   net = caffe.Net(model_proto, model_weight, caffe.TEST)
   imgs = open(imgtxt,'r').readlines()
   
   count = 0
   acc = 0
   for imgname in imgs:
      imgname,label = imgname.strip().split(' ')
      imgtype = imgname.split('.')[-1]
      if imgtype != 'png' and imgtype != 'jpg' and imgtype != 'JPG' and imgtype != 'jpeg' and imgtype != 'tif' and imgtype != 'bmp':
          print imgtype,"error"
          continue
      imgpath = imgname

      img = cv2.imread(imgpath)
      if img is None:
          print "---------img is empty---------",imgpath
          continue
  
      imgheight,imgwidth,channel = img.shape

      if enable_crop == 1:
          print "use crop"
          cropx = (imgwidth - testsize) / 2 
          cropy = (imgheight - testsize) / 2
          img = img[cropy:cropy+testsize,cropx:cropx+testsize,0:channel]
      else:
          img = cv2.resize(img,(testsize,testsize),interpolation=cv2.INTER_NEAREST)

      transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
      transformer.set_mean('data', np.array([104.008,116.669,122.675]))
      transformer.set_transpose('data', (2,0,1))
              
      out = net.forward_all(data=np.asarray([transformer.preprocess('data', img)]))
         
      result = out['classifier'][0]
      print "result=",result
      predict = np.argmax(result) 
      if str(label) == str(predict):
         acc = acc + 1
      count = count + 1
   
   print "acc=",float(acc) / float(count)
if __name__ == '__main__':
    args = parse_args()
    start_test(args.model_proto,args.model_weight,args.imgtxt,args.testsize,int(args.enable_crop))
