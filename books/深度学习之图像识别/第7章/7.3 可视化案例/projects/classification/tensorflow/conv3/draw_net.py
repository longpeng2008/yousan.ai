from dataset import *
from net_debug import simpleconv3net
import sys
import os
import cv2

txtfile = sys.argv[1]
batch_size = 64
num_classes = 2
image_size = (48,48)
learning_rate = 0.0001

debug=False

if __name__=="__main__":
    dataset = ImageData(txtfile,batch_size,num_classes,image_size)
    iterator = dataset.data.make_one_shot_iterator()
    dataset_size = dataset.dataset_size
    batch_images,batch_labels = iterator.get_next()
    Ylogits = simpleconv3net(batch_images)

    print "Ylogits size=",Ylogits.shape

    Y = tf.nn.softmax(Ylogits)
    with tf.name_scope("loss"):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=batch_labels)
        cross_entropy = tf.reduce_mean(cross_entropy)
    with tf.name_scope("acc"):
        correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(batch_labels, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

    saver = tf.train.Saver()
    in_steps = 100
    checkpoint_dir = 'checkpoints/'
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    log_dir = 'logs/'
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    with tf.Session() as sess:  
        init = tf.global_variables_initializer()
        sess.run(init)  
        steps = 10000 
        summary = tf.summary.FileWriter("output", sess.graph)
        for i in range(steps): 
            _,cross_entropy_,accuracy_,batch_images_,batch_labels_ = sess.run([train_step,cross_entropy,accuracy,batch_images,batch_labels])
            if i % in_steps == 0 :
                print i,"iterations,loss=",cross_entropy_,"acc=",accuracy_
                saver.save(sess, checkpoint_dir + 'model.ckpt', global_step=i)    
                #print "predict=",Ylogits," labels=",batch_labels

                if debug:
                    imagedebug = batch_images_[0].copy()
                    imagedebug = np.squeeze(imagedebug)
                    print imagedebug,imagedebug.shape
                    print np.max(imagedebug)
                    imagelabel = batch_labels_[0].copy()
                    print np.squeeze(imagelabel)

                    imagedebug = cv2.cvtColor((imagedebug*255).astype(np.uint8),cv2.COLOR_RGB2BGR)
                    cv2.namedWindow("debug image",0)
                    cv2.imshow("debug image",imagedebug)
                    k = cv2.waitKey(0)
                    if k == ord('q'):
                        break
        summary.close()


