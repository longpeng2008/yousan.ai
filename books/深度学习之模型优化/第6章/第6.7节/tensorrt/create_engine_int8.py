#coding:utf8

import tensorrt as trt
import os
import torch
import torch.nn.functional as F
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import ctypes
import glob,os
from PIL import Image
ctypes.pythonapi.PyCapsule_GetPointer.restype = ctypes.c_char_p
ctypes.pythonapi.PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]

# 4种不同的校正方法
#IInt8EntropyCalibrator2
#IInt8LegacyCalibrator
#IInt8EntropyCalibrator
#IInt8MinMaxCalibrator
class Calibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, stream, cache_file=""):
        trt.IInt8EntropyCalibrator2.__init__(self)       
        self.stream = stream
        self.d_input = cuda.mem_alloc(self.stream.calibration_data.nbytes)
        self.cache_file = cache_file
        stream.reset()

    def get_batch_size(self):
        return self.stream.batch_size

    def get_batch(self, names):
        #print("############################################################")
        #print(names)
        #print("############################################################")
        batch = self.stream.next_batch()
        if not batch.size:   
            return None

        cuda.memcpy_htod(self.d_input, batch)
        return [int(self.d_input)]

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)

def preprocess_img(img):
    input_size = [224,224]
    mean = (0.5,0.5,0.5)
    std = (0.5,0.5,0.5)
    img = np.asarray(img.resize((input_size[0],input_size[1]),resample=Image.NEAREST)).astype(np.float32) / 255.0
    img[:,:,] -= mean
    img[:,:,] /= std
    input_data = img.transpose([2, 0, 1])
    return input_data

class DataLoader:
    def __init__(self,batch,batchsize,img_height,img_width,images_dir):
        self.index = 0
        self.length = batch
        self.batch_size = batchsize
        # self.img_list = [i.strip() for i in open('calib.txt').readlines()]
        self.img_list = glob.glob(os.path.join(images_dir, "**/*.jpg"),recursive=True)
        print('found all {} images to calib.'.format(len(self.img_list)))
        assert len(self.img_list) >= self.batch_size * self.length, '{} must contains more than '.format(images_dir) + str(self.batch_size * self.length) + ' images to calib'
        self.calibration_data = np.zeros((self.batch_size,3,img_height,img_width), dtype=np.float32)

    def reset(self):
        self.index = 0

    def next_batch(self):
        if self.index < self.length:
            for i in range(self.batch_size):
                assert os.path.exists(self.img_list[i + self.index * self.batch_size]), 'not found!!'
                img = Image.open(self.img_list[i + self.index * self.batch_size]).convert('RGB')
                img = preprocess_img(img)
                self.calibration_data[i] = img

            self.index += 1

            return np.ascontiguousarray(self.calibration_data, dtype=np.float32)
        else:
            return np.array([])

    def __len__(self):
        return self.length

# 创建推理参数
BATCH = 125 # 总batch数量
BATCH_SIZE = 8 # 每一个batch图片数
CALIB_IMG_DIR = 'GHIM-20' # 校准数据集
IMG_HEIGHT = 224 # 模型输入高
IMG_WIDTH = 224 # 模型输入宽
model_path = 'simpleconv5.onnx' # 输入float32模型
calibration_table = 'simpleconv5.cache'


# 创建日志和builder,network
logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)
builder.int8_mode = True
builder.fp16_mode = False
calibration_stream = DataLoader(BATCH,BATCH_SIZE,IMG_HEIGHT,IMG_WIDTH,CALIB_IMG_DIR)
builder.int8_calibrator = Calibrator(calibration_stream,calibration_table) 
print("int8 model enabled")

## 从ONNX格式模型创建引擎
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, logger)
success = parser.parse_from_file(model_path)

# 创建引擎
engine = builder.build_cuda_engine(network)

# 序列化engine并且写入文件中
with open("simpleconv5_int8.trt","wb") as f:
    f.write(engine.serialize())

