#!/usr/bin/env python3
from chainer import cuda, Function, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
import dataset
import pickle
import os.path
class MyModel(Chain):
	def __init__(self):
		super(MyModel, self).__init__()
		with self.init_scope():
			self.conv1 = L.Convolution2D(
				in_channels=3, out_channels=12, ksize=3, stride=2)
			self.bn1 = L.BatchNormalization(12)
			self.conv2 = L.Convolution2D(
				in_channels=12, out_channels=24, ksize=3, stride=2)
			self.bn2 = L.BatchNormalization(24)
			self.conv3 = L.Convolution2D(
				in_channels=24, out_channels=48, ksize=3, stride=2)
			self.bn3 = L.BatchNormalization(48)
			self.fc1 = L.Linear(None, 128)
			self.fc2 = L.Linear(128, 2)
	def __call__(self,x):
		return self.forward(x)
	def forward(self, x):
		h1 = F.relu(self.conv1(x))
		h2 = F.relu(self.conv2(h1))
		h3 = F.relu(self.conv3(h2))
		h4 = F.relu(self.fc1(h3))
		x = self.fc2(h4)
		return (x)
if __name__=='__main__':
	model = L.Classifier(MyModel())
	if os.path.isfile('./dataset.pickle'):
		print("dataset.pickle is exist. loading...")
		with open('./dataset.pickle', mode='rb') as f:
			train, test = pickle.load(f)
			print("Loaded")
	else:
		datasets = dataset.Dataset("mouth")
		train, test = datasets.get_dataset()
		with open('./dataset.pickle', mode='wb') as f:
			pickle.dump((train, test), f)
			print("saving train and test...")
	optimizer = optimizers.MomentumSGD(lr=0.001, momentum=0.9)
	optimizer.setup(model)
	train_iter = iterators.SerialIterator(train, 64)
	test_iter = iterators.SerialIterator(test, 64, repeat=False, shuffle=True)

	updater = training.StandardUpdater(train_iter, optimizer, device=-1)
	trainer = training.Trainer(updater, (800, 'epoch'), out='{}_model_result'.format(MyModel.__class__.__name__))
	trainer.extend(extensions.dump_graph("main/loss"))
	trainer.extend(extensions.Evaluator(test_iter, model, device=-1))
	trainer.extend(extensions.LogReport())
	trainer.extend(extensions.PrintReport( ['epoch', 'main/loss', 'validation/main/loss', 'main/accuracy', 'validation/main/accuracy']))
	trainer.extend(extensions.PlotReport(['main/loss', 'validation/main/loss'], x_key='epoch', file_name='loss.png'))
	trainer.extend(extensions.PlotReport(['main/accuracy', 'validation/main/accuracy'], x_key='epoch', file_name='accuracy.png'))
	trainer.extend(extensions.ProgressBar())
	trainer.run()
	print("Learn END")

