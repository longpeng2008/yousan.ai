#coding:utf8

import tensorrt as trt

# 创建日志和builder,network
logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

## 从ONNX格式模型创建引擎
parser = trt.OnnxParser(network, logger)
model_path = "simpleconv5.onnx"
success = parser.parse_from_file(model_path)

# 创建引擎
engine = builder.build_cuda_engine(network)

# 序列化engine并且写入文件中
with open("simpleconv5.trt","wb") as f:
    f.write(engine.serialize())

