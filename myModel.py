import tensorflow as tf
from datetime import datetime

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('train_dir', 'data/train_32x32.mat', 'Directory of training data')
flags.DEFINE_string('test_dir', 'data/test_32x32.mat', 'Directory of test data')
flags.DEFINE_string('checkpoint_dir', 'checkpoints', 'Directory of checkpoint files')
flags.DEFINE_string('log_dir', 'tensorBoard_log', 'Directory of tensor board log')
flags.DEFINE_string('gpu', '0', 'Select gpu')
flags.DEFINE_float('learning_rate', 0.01, 'Learning rate')
flags.DEFINE_integer('batch_size', 16, 'Batch size')
flags.DEFINE_integer('training_epochs', 10, 'The number of training epoch')


# Define the architecture of my model

class model:

    def __init__(self, sess, name):
        """
        Model class initializer
        :param sess: Take tensorFlow session as input
        :param name: The name of the model
        """
        print "Initializing model..."
        self.sess = sess
        self.name = name
        self._build_net()
        self.writer = tf.summary.FileWriter(FLAGS.log_dir)
        self.writer.add_graph(self.sess.graph)
        self.saver = tf.train.Saver()

        # Training variables
        self.global_step = 0
        self.last_epoch = tf.Variable(0, name="last_epoch")
        self.avg_cost = 0
        tf.summary.scalar("avg_cost", self.avg_cost)
        self.summary = tf.summary.merge_all()

    def _build_net(self):
        """
        Define the architecture of the model
        """

        with tf.variable_scope(self.name):
            # Input place holders
            self.training = tf.placeholder(tf.bool)
            self.X = tf.placeholder(tf.float32, [None, 32, 32, 3])
            self.Y = tf.placeholder(tf.int32, [None, 10])

            with tf.variable_scope("Conv_layers"):
                conv1 = tf.layers.conv2d(inputs=self.X, filters=16, kernel_size=[3, 3], padding="SAME", activation=tf.nn.relu)
                conv2 = tf.layers.conv2d(inputs=conv1, filters=16, kernel_size=[3, 3], padding="SAME", activation=tf.nn.relu)
                pool1 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

                conv3 = tf.layers.conv2d(inputs=pool1, filters=32, kernel_size=[3, 3], padding="SAME", activation=tf.nn.relu)
                conv4 = tf.layers.conv2d(inputs=conv3, filters=32, kernel_size=[3, 3], padding="SAME", activation=tf.nn.relu)
                pool2 = tf.nn.max_pool(conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

                conv5 = tf.layers.conv2d(inputs=pool2, filters=64, kernel_size=[3, 3], padding="SAME", activation=tf.nn.relu)
                conv6 = tf.layers.conv2d(inputs=conv5, filters=64, kernel_size=[3, 3], padding="SAME", activation=tf.nn.relu)
                pool3 = tf.nn.max_pool(conv6, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

                tf.summary.tensor_summary("conv1", conv1)
                tf.summary.tensor_summary("conv2", conv2)
                tf.summary.tensor_summary("conv3", conv3)
                tf.summary.tensor_summary("conv4", conv4)
                tf.summary.tensor_summary("conv5", conv5)
                tf.summary.tensor_summary("conv6", conv6)

            with tf.variable_scope("Dense_layers"):
                flat = tf.reshape(conv6, [-1, 8 * 8 * 64])
                fc1 = tf.layers.dense(flat, 1024)
                fc2 = tf.layers.dense(fc1, 256)
                self.logits = tf.layers.dense(fc2, 10)

                tf.summary.tensor_summary("fc1", fc1)
                tf.summary.tensor_summary("fc2", fc2)
                tf.summary.tensor_summary("logits", self.logits)

            # Define cost function and optimizer
            self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y))
            self.optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(self.cost)

            tf.summary.scalar("Cost", self.cost)

    def train(self, inputs, labels):
        start_epoch = self.sess.run(self.last_epoch)
        print datetime.today().strftime("[%Y.%m.%d|%H:%M:%S] ") + "Start learning from epoch:", start_epoch

        for epoch in range(start_epoch, FLAGS.training_epochs):
            print datetime.today().strftime("[%Y.%m.%d|%H:%M:%S] ") + "Starting epoch:", epoch
            self.avg_cost = 0

            for batch in range(len(labels) / FLAGS.batch_size):
                loss, summary, _ = self.sess.run([self.cost, self.summary, self.optimizer], feed_dict={self.X: inputs[batch*FLAGS.batch_size:(batch+1)*FLAGS.batch_size, :, :, :],
                                                                                                       self.Y: labels[batch*FLAGS.batch_size:(batch+1)*FLAGS.batch_size, :]})
                self.writer.add_summary(summary, global_step=self.global_step)
                self.avg_cost += loss / len(labels)
                self.global_step += 1

                if batch % 300 == 0:
                    print datetime.today().strftime("[%Y.%m.%d|%H:%M:%S] ") + 'Batch: %d, Loss: %f, Avg: %f' \
                                                                              % (self.global_step, loss, self.avg_cost)

                if self.global_step % 1250 == 0:
                    self.save()

        print datetime.today().strftime("[%Y.%m.%d|%H:%M:%S] ") + "Learning finished!"
        self.save()

    def predict(self, x_test):
        return self.sess.run(tf.argmax(self.logits), feed_dict={self.X: x_test})

    def save(self):
        print datetime.today().strftime("[%Y.%m.%d|%H:%M:%S] ") + "Saving model at " + FLAGS.checkpoint_dir
        self.saver.save(self.sess, FLAGS.checkpoint_dir + '/model', global_step=self.global_step)

    def restore(self):
        checkpoint = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)

        if checkpoint and checkpoint.model_checkpoint_path:
            try:
                self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
                print datetime.today().strftime("[%Y.%m.%d|%H:%M:%S] ") + "Model is successfully loaded from ", checkpoint.model_checkpoint_path
            except:
                print datetime.today().strftime("[%Y.%m.%d|%H:%M:%S] ") + "Error on loading model weights"
        else:
            print datetime.today().strftime("[%Y.%m.%d|%H:%M:%S] ") + "Could not find checkpoint"
