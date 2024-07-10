#coding:utf8
# 导入TensorRT，CUDA，PIL等相关库
import torch
import tensorrt as trt
import numpy as np
from PIL import Image
import pycuda.driver as cuda
import pycuda.autoinit #pycuda.driver初始化
import sys
import time
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# 日志接口，TensorRT通过该接口报告错误、警告和信息性消息
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# 从文件中读取engine并且反序列化
modelpath = sys.argv[1]
start_Deserialize = time.time()
with open(modelpath,"rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
	engine = runtime.deserialize_cuda_engine(f.read())
end_Deserialize = time.time()
print('deserialize use time='+str((end_Deserialize-start_Deserialize)*1000)+' ms')

# 创建执行上下文		
context = engine.create_execution_context()

# 使用pycuda输入和输出分配内存
# engine有一个输入binding_index=0和一个输出binding_index=1
h_input = cuda.pagelocked_empty(trt.volume(context.get_binding_shape(0)), dtype=np.float32) # 输入CPU内存
h_output = cuda.pagelocked_empty(trt.volume(context.get_binding_shape(1)), dtype=np.float32) # 输出CPU内存
d_input = cuda.mem_alloc(h_input.nbytes) # 输入GPU内存
d_output = cuda.mem_alloc(h_output.nbytes) # 输出GPU内存
print('binding size='+str(context.get_binding_shape(0)))

# 图片预处理
start_preprocessing = time.time()
imgpath = 'data/sample.jpg'
input_size = [224,224]
img = Image.open(imgpath)
mean = (0.5,0.5,0.5)
std = (0.5,0.5,0.5)
img = np.asarray(img.resize((input_size[0],input_size[1]),resample=Image.NEAREST)).astype(np.float32) / 255.0
img[:,:,] -= mean
img[:,:,] /= std
input_data = img.transpose([2, 0, 1]).ravel()
# print('input data shape='+str(input_data.shape))
np.copyto(h_input,input_data) 
end_preprocessing = time.time()
print('image preprocessing use time='+str((end_preprocessing-start_preprocessing)*1000)+' ms')

# 创建一个流,在其中复制输入/输出并且运行推理
stream = cuda.Stream()

# 将输入数据转换到GPU上
start_CPU2GPU = time.time()
cuda.memcpy_htod_async(d_input, h_input, stream)
#print('input data='+str(h_input))
end_CPU2GPU = time.time()
print('cpu2gpu use time='+str((end_CPU2GPU-start_CPU2GPU)*1000)+' ms')

# 运行推理
context.execute_v2(bindings=[int(d_input),int(d_output)]) ## GPU的预热

stream.synchronize() # GPU的异步处理
epoch = 100 # 推理轮次
start_Inference = time.time()
for i in range(0,epoch):
    context.execute_v2(bindings=[int(d_input),int(d_output)])
    #context.execute_async_v2(bindings=[int(d_input),int(d_output)], stream_handle=stream.handle)
end_Inference = time.time()
print('Inference use time='+str((end_Inference-start_Inference)*1000/epoch)+' ms')
stream.synchronize() # GPU的异步处理

# 从GPU上传输预测值
start_GPU2CPU = time.time()
cuda.memcpy_dtoh_async(h_output, d_output, stream)

# 同步流
end_GPU2CPU = time.time()
print('gpu2cpu use time='+str((end_GPU2CPU-start_GPU2CPU)*1000)+' ms')

print('output data shape='+str(h_output.shape))
#print('output data='+str(h_output))
pred1 = np.argmax(h_output)
print("=======pred 1===========", pred1)
