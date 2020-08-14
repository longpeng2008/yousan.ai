#coding:utf8

# Copyright 2019 longpeng2008. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# If you find any problem,please contact us
#
#     longpeng2008to2012@gmail.com 
#
# or create issues
# =============================================================================
import lasagne
import theano
import theano.tensor as T
import sys
import numpy as np

import matplotlib.pyplot as plt

from net import simpleconv3
from dataset import Dataset

input_var = T.tensor4('X')
target_var = T.ivector('y')

network = simpleconv3(input_var)

prediction = lasagne.layers.get_output(network)
loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
loss = loss.mean()

#loss = loss.mean() + 1e-4 * lasagne.regularization.regularize_network_params(
#        network, lasagne.regularization.l2)

test_prediction = lasagne.layers.get_output(network, deterministic=True) 
test_loss = lasagne.objectives.categorical_crossentropy(test_prediction, target_var) 
test_loss = test_loss.mean()
test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var), dtype=theano.config.floatX)

# create parameter update expressions
params = lasagne.layers.get_all_params(network, trainable=True)
updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=0.001,
                                            momentum=0.9)

# compile train function that updates parameters and returns train loss
train_fn = theano.function([input_var, target_var], loss, updates=updates)
val_fn = theano.function([input_var, target_var], [test_loss, test_acc])
predict_fn = theano.function([input_var], T.argmax(test_prediction, axis=1))


# prepare mydataset
mydataset = Dataset(sys.argv[1],48,48)

train_losses = []
val_losses = []
val_accs = []
num_of_epoches = 100

for epoch in range(num_of_epoches):
    train_loss = 0
    train_data =  mydataset.iterate_minibatches(mydataset.imagetraindatas,mydataset.labeltraindatas,16,True)
    train_batches = 0
    for input_batch, target_batch in train_data:
        train_loss += train_fn(input_batch, target_batch)
        train_batches += 1
    print("Epoch %d: Train Loss %g" % (epoch + 1, train_loss / train_batches))
    train_losses.append(train_loss / train_batches)
   
    val_loss = 0
    val_acc = 0
    val_batches = 0
    val_data =  mydataset.iterate_minibatches(mydataset.imagevaldatas,mydataset.labelvaldatas,mydataset.getvallen(),False)
    for val_batch,target_batch in val_data:
        tmp_val_loss, tmp_val_acc = val_fn(val_batch, target_batch)
        val_loss += tmp_val_loss
        val_acc += tmp_val_acc
        val_batches += 1
        test_prediction = lasagne.layers.get_output(network, deterministic=True)
        val_predict = predict_fn(val_batch)


    print("Epoch %d: Val Loss %g" % (epoch + 1, val_loss / val_batches))
    print("Epoch %d: Val Acc %g" % (epoch + 1, val_acc / val_batches))
    val_losses.append(val_loss / val_batches)
    val_accs.append(val_acc / val_batches)

x = list(xrange(num_of_epoches))
plt.plot(x,train_losses,'r-o')
plt.plot(x,val_losses,'k-o')
plt.legend(('train','val'))
plt.title('train_val_loss')
plt.savefig('loss.png')
plt.show()

plt.plot(x,val_accs)
plt.savefig('acc.png')
plt.show()

