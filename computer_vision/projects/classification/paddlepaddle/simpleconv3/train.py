#coding:utf8

# Copyright 2019 longpeng2008. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# If you find any problem,please contact us
#
#     longpeng2008to2012@gmail.com 
#
# or create issues
# =============================================================================
import os
import sys
import paddle.v2 as paddle
from dataset import Dataset
from net import simplenet
from visualdl import LogWriter

class SimpleConv3:
    def __init__(self):
        paddle.init(use_gpu=False, trainer_count=2)

    def get_parameters(self, parameters_path=None, cost=None):
        if not parameters_path:
            if not cost:
                raise NameError('请输入cost参数')
            else:
                parameters = paddle.parameters.create(cost)
                print "cost"
                return parameters
        else:
            try:
                with open(parameters_path, 'r') as f:
                    parameters = paddle.parameters.Parameters.from_tar(f)
                return parameters
            except Exception as e:
                raise NameError("你的参数文件错误,具体问题是:%s" % e)

    # ***********************获取训练器***************************************
    # datadim 数据大小
    def get_trainer(self, datadim, type_size, parameters_path):
        label = paddle.layer.data(name="label",
                                  type=paddle.data_type.integer_value(type_size))
        out = simplenet(datadim=datadim, type_size=type_size)
        print "out=",out
        cost = paddle.layer.classification_cost(input=out, label=label)

        if not parameters_path:
            parameters = self.get_parameters(cost=cost)
        else:
            parameters = self.get_parameters(parameters_path=parameters_path)
        optimizer = paddle.optimizer.Momentum(
            momentum=0.9,
            regularization=paddle.optimizer.L2Regularization(rate=0.0005 * 128),
            learning_rate=0.001,
            learning_rate_decay_a=0.1,
            learning_rate_decay_b=128000 * 35,
            learning_rate_schedule="discexp", )

        trainer = paddle.trainer.SGD(cost=cost,
                                     parameters=parameters,
                                     update_equation=optimizer)
        return trainer

    # ***********************开始训练***************************************
    def start_trainer(self, trainer, num_passes, save_parameters_name, trainer_reader, test_reader):
        reader = paddle.batch(reader=paddle.reader.shuffle(reader=trainer_reader,
                                                           buf_size=50000),
                              batch_size=128)
        father_path = save_parameters_name[:save_parameters_name.rfind("/")]
        if not os.path.exists(father_path):
            os.makedirs(father_path)

        # 指定每条数据和padd.layer.data的对应关系
        feeding = {"image": 0, "label": 1}

        # 定义训练事件
    	step = 0
        def event_handler(event):
            if isinstance(event, paddle.event.EndIteration):
                print "look event",event
                if event.batch_id % 1 == 0:
                    print "\nPass %d, Batch %d, Cost %f, Error %s" % (
                        event.pass_id, event.batch_id, event.cost, event.metrics['classification_error_evaluator'])
                else:
                    sys.stdout.write('.')
                    sys.stdout.flush()

            # 每一轮训练完成之后
            if isinstance(event, paddle.event.EndPass):
                # 保存训练好的参数
                with open(save_parameters_name, 'w') as f:
                    trainer.save_parameter_to_tar(f)

                # 测试准确率
                result = trainer.test(reader=paddle.batch(reader=test_reader,batch_size=64),feeding=feeding)
                print "\nTest with Pass %d, Classification_Error %s" % (
                event.pass_id, result.metrics['classification_error_evaluator'])

            	#loss_scalar.add_record(step, event.cost)
		#step +=1

        trainer.train(reader=reader,
                      num_passes=num_passes,
                      event_handler=event_handler,
                      feeding=feeding)


if __name__ == '__main__':
    type_size = 2
    resizesize = 60
    cropsize = 48
    parameters_path = "./snaps/model.tar"
    datadim = 3 * cropsize * cropsize
    clcmodel = SimpleConv3()

    myReader = Dataset(cropsize=cropsize,resizesize=resizesize)
    trainer = clcmodel.get_trainer(datadim=datadim, type_size=type_size, parameters_path=None)
    trainer_reader = myReader.train_reader(train_list='./all_shuffle_train.txt')
    test_reader = myReader.test_reader(test_list='./all_shuffle_val.txt')

    clcmodel.start_trainer(trainer=trainer, num_passes=100, save_parameters_name=parameters_path,
                             trainer_reader=trainer_reader, test_reader=test_reader)
