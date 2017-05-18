import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np
import myModel

config = tf.ConfigProto(allow_soft_placement=True)
sess = tf.Session(config=config)
m1 = myModel.model(sess, "toy")
sess.run(tf.global_variables_initializer())

train_data = sio.loadmat(myModel.FLAGS.train_dir)
test_data = sio.loadmat(myModel.FLAGS.test_dir)

img_train = np.array(train_data['X'])
label_train = np.array(train_data['y'])
img_test = np.array(test_data['X'])
label_test = np.array(test_data['y'])

for i in range(len(label_train)):
    if label_train[i] == [10]:
        label_train[i] = [0]

for i in range(len(label_test)):
    if label_test[i] == [10]:
        label_test[i] = [0]

img_train = tf.reshape(img_train, [len(label_train), 32, 32, 3])
img_test = tf.reshape(img_test, [len(label_test), 32, 32, 3])
label_train = tf.one_hot(tf.reshape(label_train, [len(label_train)]), 10)
label_test = tf.one_hot(tf.reshape(label_test, [len(label_test)]), 10)

img_train, img_test, label_train, label_test = sess.run([img_train, img_test, label_train, label_test])

print "Data loaded"

m1.train(img_train, label_train)
sess.close()