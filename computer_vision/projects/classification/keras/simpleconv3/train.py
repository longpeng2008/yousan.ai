from __future__ import print_function, division

import os
from keras.models import *
from keras.optimizers import SGD
from keras.losses import binary_crossentropy
from keras.metrics import binary_accuracy
from keras.callbacks import TensorBoard
from net import simpleconv3
from keras.preprocessing.image import ImageDataGenerator


'''
datas/
    train/
        left/
            *.jpg
            *.jpg
            ...
        right/
            *.jpg
            *.jpg
    val/
        right/
            *.jpg
            *.jpg
        left/
            *.jpg
            *.jpg
'''

def train_model(model, loss, metrics, optimizer, epochs=25):

    model.compile(loss=loss,
                  optimizer=optimizer,
                  metrics=metrics)
    model.summary()

    history = model.fit_generator(
        train_generator,
        steps_per_epoch=num_train_samples // batch_size,
        epochs=epochs,
        callbacks=callbacks,
        validation_data=val_generator,
        validation_steps=num_val_samples // batch_size)
    return model


if __name__ == '__main__':
    # dimensions of our images.
    img_width, img_height = 48, 48
    num_epochs = 25
    batch_size = 16

    train_data_dir = 'data/train/'
    validation_data_dir = 'data/val'

    # this is the augmentation configuration we will use for training
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
    # this is the augmentation configuration use for testing only rescaling
    val_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size)

    val_generator = val_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size)

    num_train_samples = train_generator.samples
    num_val_samples = val_generator.samples

    tensorboard = TensorBoard(log_dir=('./logs'))
    callbacks = []
    callbacks.append(tensorboard)
    model = simpleconv3()
    loss = binary_crossentropy
    metrics = [binary_accuracy]
    optimizer = SGD(lr=0.001, decay=1e-6, momentum=0.9)

    model = train_model(model, loss, metrics,  optimizer, num_epochs)
    if not os.path.exists('models'):
        os.mkdir('models')
    model_json = model.to_json()
    with open('models/model.json', 'w') as json_file:
        json_file.write(model_json)
    model.save_weights('models/model.h5')
