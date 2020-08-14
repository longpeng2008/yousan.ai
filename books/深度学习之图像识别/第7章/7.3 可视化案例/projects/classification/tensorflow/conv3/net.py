import tensorflow as tf

debug=True

def simpleconv3net(x):
    x_shape = tf.shape(x)
    with tf.name_scope("simpleconv3"):
        with tf.variable_scope("conv3_net"):
            conv1 = tf.layers.conv2d(x, name="conv1", filters=12,kernel_size=[3,3], strides=(2,2), activation=tf.nn.relu,kernel_initializer=tf.contrib.layers.xavier_initializer(),bias_initializer=tf.contrib.layers.xavier_initializer())
            bn1 = tf.layers.batch_normalization(conv1, training=True, name='bn1')
            conv2 = tf.layers.conv2d(bn1, name="conv2", filters=24,kernel_size=[3,3], strides=(2,2), activation=tf.nn.relu,kernel_initializer=tf.contrib.layers.xavier_initializer(),bias_initializer=tf.contrib.layers.xavier_initializer())
            bn2 = tf.layers.batch_normalization(conv2, training=True, name='bn2')
            conv3 = tf.layers.conv2d(bn2, name="conv3", filters=48,kernel_size=[3,3], strides=(2,2), activation=tf.nn.relu,kernel_initializer=tf.contrib.layers.xavier_initializer(),bias_initializer=tf.contrib.layers.xavier_initializer())
            bn3 = tf.layers.batch_normalization(conv3, training=True, name='bn3')
            conv3_flat = tf.reshape(bn3, [-1, 5 * 5 * 48])
            dense = tf.layers.dense(inputs=conv3_flat, units=128, activation=tf.nn.relu,name="dense",kernel_initializer=tf.contrib.layers.xavier_initializer())
            logits= tf.layers.dense(inputs=dense, units=2, activation=tf.nn.relu,name="logits",kernel_initializer=tf.contrib.layers.xavier_initializer())
        
            if debug:
                print "x size=",x.shape
                print "relu_conv1 size=",conv1.shape
                print "relu_conv2 size=",conv2.shape
                print "relu_conv3 size=",conv3.shape
                print "dense size=",dense.shape
                print "logits size=",logits.shape

    return logits
