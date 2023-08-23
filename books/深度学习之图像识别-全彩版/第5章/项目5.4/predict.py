'''
predict.py有几个注意点
1、如果想要保存，利用r_image.save("img.jpg")即可保存。
2、如果想要获得框的坐标，可以进入detect_image函数，读取top,left,bottom,right这四个值。
3、如果想要截取下目标，可以利用获取到的top,left,bottom,right这四个值在原图上利用矩阵的方式进行截取。
'''
from PIL import Image
import os
from yolo import YOLO

yolo = YOLO()

from tqdm import tqdm
dir_origin_path = 'img/src'
dir_save_path = 'img/result'
img_names = os.listdir(dir_origin_path)
for img_name in tqdm(img_names):
    if img_name.lower().endswith(
            ('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
        image_path = os.path.join(dir_origin_path, img_name)
        image = Image.open(image_path) ## 读取图片
        r_image = yolo.detect_image(image) ## 调用detect_image函数
        r_image.show()
        if not os.path.exists(dir_save_path):
            os.makedirs(dir_save_path)
        r_image.save(os.path.join(dir_save_path, img_name.replace(".jpg", ".png")), quality=95, subsampling=0)

