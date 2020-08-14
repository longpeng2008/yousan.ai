import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(12, (3,3), activation='relu', input_shape=(48, 48, 3),strides=(2, 2), padding='same'),
    tf.keras.layers.BatchNormalization(axis=3),
    tf.keras.layers.Conv2D(24, (3,3), activation='relu',strides=(2, 2), padding='same'),
    tf.keras.layers.BatchNormalization(axis=3),
    tf.keras.layers.Conv2D(48, (3,3), activation='relu',strides=(2, 2), padding='same'),
    tf.keras.layers.BatchNormalization(axis=3),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy',
               optimizer = SGD(lr=0.001, decay=1e-6, momentum=0.9),
              metrics=['acc'])

train_datagen = ImageDataGenerator(rescale=1/255, shear_range=0.2,zoom_range=0.2, horizontal_flip=True)
validation_datagen = ImageDataGenerator(rescale=1/255)
train_generator = train_datagen.flow_from_directory(
        r"D://Learning//tensorflow_2.0//data//train",  # 训练集的根目录
        target_size=(48, 48),  # 所有图像的分辨率将被调整为48x48
        batch_size=32,			 # 每次输入32个图像
        # 类别模式设为二分类
        class_mode='binary')

# 对验证集做同样的操作
validation_generator = validation_datagen.flow_from_directory(
      r"D://Learning//tensorflow_2.0//data//val",
        target_size=(48, 48),
       batch_size=16,
        class_mode='binary')
history = model.fit_generator(
      train_generator,
      steps_per_epoch=28,
      epochs=50,
      verbose=1,
      validation_data = validation_generator,
    callbacks=[TensorBoard(log_dir=(r"D:\Learning\logs"))],
      validation_steps=6)
