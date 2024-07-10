#coding:utf8
import torch
import torchvision
from torchsummary import summary
import time
import cv2
import onnxruntime
import numpy as np
import PIL.Image as Image
import os,glob
from simpleconv5 import simpleconv5
torch.manual_seed(0)
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit #pycuda.driver初始化
import sys
import time
import os,glob
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

imgdir = "GHIM-20"
##-----------------------fp32 time------------------------##
# 日志接口，TensorRT通过该接口报告错误、警告和信息性消息
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
# 从文件中读取engine并且反序列化
modelpath = "simpleconv5.trt"
with open(modelpath,"rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
	engine = runtime.deserialize_cuda_engine(f.read())
# 创建执行上下文		
context = engine.create_execution_context()
# 使用pycuda输入和输出分配内存
# engine有一个输入binding_index=0和一个输出binding_index=1
h_input = cuda.pagelocked_empty(trt.volume(context.get_binding_shape(0)), dtype=np.float32) # 输入CPU内存
h_output = cuda.pagelocked_empty(trt.volume(context.get_binding_shape(1)), dtype=np.float32) # 输出CPU内存
d_input = cuda.mem_alloc(h_input.nbytes) # 输入GPU内存
d_output = cuda.mem_alloc(h_output.nbytes) # 输出GPU内存
# 创建一个流,在其中复制输入/输出并且运行推理
stream = cuda.Stream()

# 图片预处理
acc = 0.0
nums = 0.0
stream.synchronize() # GPU的异步处理
start_Inference = time.time()
for imgpath in glob.glob(os.path.join(imgdir, "**/*.jpg"),recursive=True):
    img = Image.open(imgpath).convert('RGB')
    input_size = [224,224]
    mean = (0.5,0.5,0.5)
    std = (0.5,0.5,0.5)
    img = np.asarray(img.resize((input_size[0],input_size[1]),resample=Image.NEAREST)).astype(np.float32) / 255.0
    img[:,:,] -= mean
    img[:,:,] /= std
    input_data = img.transpose([2, 0, 1]).ravel()
    np.copyto(h_input,input_data) 
    cuda.memcpy_htod_async(d_input, h_input, stream)
    context.execute_v2(bindings=[int(d_input),int(d_output)]) ## GPU的预热
    cuda.memcpy_dtoh_async(h_output, d_output, stream)
    pred1 = np.argmax(h_output)
    prob1 = h_output[pred1]
    label = int(imgpath.split('/')[-2])
    if label == pred1:
        acc += 1.0
    nums += 1.0

stream.synchronize() # GPU的异步处理
end_Inference = time.time()
print('TensorRT fp32 Inference use time='+str((end_Inference-start_Inference)*1000/nums)+' ms')
print("acc=",acc/nums)

##-----------------------fp16 time------------------------##
# 日志接口，TensorRT通过该接口报告错误、警告和信息性消息
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
# 从文件中读取engine并且反序列化
modelpath = "simpleconv5_fp16.trt"
with open(modelpath,"rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
	engine = runtime.deserialize_cuda_engine(f.read())
# 创建执行上下文		
context = engine.create_execution_context()
# 使用pycuda输入和输出分配内存
# engine有一个输入binding_index=0和一个输出binding_index=1
h_input = cuda.pagelocked_empty(trt.volume(context.get_binding_shape(0)), dtype=np.float32) # 输入CPU内存
h_output = cuda.pagelocked_empty(trt.volume(context.get_binding_shape(1)), dtype=np.float32) # 输出CPU内存
d_input = cuda.mem_alloc(h_input.nbytes) # 输入GPU内存
d_output = cuda.mem_alloc(h_output.nbytes) # 输出GPU内存
# 创建一个流,在其中复制输入/输出并且运行推理
stream = cuda.Stream()

# 图片预处理
acc = 0.0
nums = 0.0
stream.synchronize() # GPU的异步处理
start_Inference = time.time()
for imgpath in glob.glob(os.path.join(imgdir, "**/*.jpg"),recursive=True):
    img = Image.open(imgpath).convert('RGB')
    input_size = [224,224]
    mean = (0.5,0.5,0.5)
    std = (0.5,0.5,0.5)
    img = np.asarray(img.resize((input_size[0],input_size[1]),resample=Image.NEAREST)).astype(np.float32) / 255.0
    img[:,:,] -= mean
    img[:,:,] /= std
    input_data = img.transpose([2, 0, 1]).ravel()
    np.copyto(h_input,input_data) 
    cuda.memcpy_htod_async(d_input, h_input, stream)
    context.execute_v2(bindings=[int(d_input),int(d_output)]) ## GPU的预热
    cuda.memcpy_dtoh_async(h_output, d_output, stream)
    pred1 = np.argmax(h_output)
    prob1 = h_output[pred1]
    label = int(imgpath.split('/')[-2])
    if label == pred1:
        acc += 1.0
    nums += 1.0

stream.synchronize() # GPU的异步处理
end_Inference = time.time()
print('TensorRT fp16 Inference use time='+str((end_Inference-start_Inference)*1000/nums)+' ms')
print("acc=",acc/nums)


##-----------------------int8 time------------------------##
# 日志接口，TensorRT通过该接口报告错误、警告和信息性消息
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
# 从文件中读取engine并且反序列化
modelpath = "simpleconv5_int8.trt"
with open(modelpath,"rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
	engine = runtime.deserialize_cuda_engine(f.read())
# 创建执行上下文		
context = engine.create_execution_context()
# 使用pycuda输入和输出分配内存
# engine有一个输入binding_index=0和一个输出binding_index=1
h_input = cuda.pagelocked_empty(trt.volume(context.get_binding_shape(0)), dtype=np.float32) # 输入CPU内存
h_output = cuda.pagelocked_empty(trt.volume(context.get_binding_shape(1)), dtype=np.float32) # 输出CPU内存
d_input = cuda.mem_alloc(h_input.nbytes) # 输入GPU内存
d_output = cuda.mem_alloc(h_output.nbytes) # 输出GPU内存
# 创建一个流,在其中复制输入/输出并且运行推理
stream = cuda.Stream()

# 图片预处理
acc = 0.0
nums = 0.0
stream.synchronize() # GPU的异步处理
start_Inference = time.time()
for imgpath in glob.glob(os.path.join(imgdir, "**/*.jpg"),recursive=True):
    img = Image.open(imgpath).convert('RGB')
    input_size = [224,224]
    mean = (0.5,0.5,0.5)
    std = (0.5,0.5,0.5)
    img = np.asarray(img.resize((input_size[0],input_size[1]),resample=Image.NEAREST)).astype(np.float32) / 255.0
    img[:,:,] -= mean
    img[:,:,] /= std
    input_data = img.transpose([2, 0, 1]).ravel()
    np.copyto(h_input,input_data) 
    cuda.memcpy_htod_async(d_input, h_input, stream)
    context.execute_v2(bindings=[int(d_input),int(d_output)]) ## GPU的预热
    cuda.memcpy_dtoh_async(h_output, d_output, stream)
    pred1 = np.argmax(h_output)
    prob1 = h_output[pred1]
    label = int(imgpath.split('/')[-2])
    if label == pred1:
        acc += 1.0
    nums += 1.0

stream.synchronize() # GPU的异步处理
end_Inference = time.time()
print('TensorRT int8 Inference use time='+str((end_Inference-start_Inference)*1000/nums)+' ms')
print("acc=",acc/nums)


