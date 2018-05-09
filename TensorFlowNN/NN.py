
import tensorflow as tf

# the sample data of the input is tensorflow.examples.tutorials.mnist.input_data
# 
# [the digit-recognition]
# 
# each input is an image of 28*28  = 784
# while output is an value of 10 (0,1,...9)

class NN():

    def __init__(self, learning_rate=1e-2, max_iterators=200, batch_size=200):
        self.learning_rate = leaning_rate
        self.max_iterators = max_iterators
        self.batch_size = batch_size
        self.x = tf.placeholder(tf.float32, [None, 784])
        self.y = tf.placeholder(tf.float32, [None, 10])
        # place holder: This allows us to change the images that are input to the TensorFlow graph. 
        # This is a so-called tensor, which just means that it is a multi-dimensional vector or matrix. 
        # The data-type is set to float32 and the shape is set to [None, img_size_flat], 
        # where None means that the tensor may hold an arbitrary number of images 
        # with each image being a vector of length img_size_flat.

    def train_step(self, logits, labels):
        # logits: actual outputs
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
        return optimizer.minimize(loss)

    def train(self):
        prediction = self.inference()
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(self.y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        
        train_op = self.train_step(prediction, self.y)
        mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
        test_images, test_labels = mnist.test.images[:self.batch_size], mnist.test.labels[:self.batch_size]

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for step in range(1, self.max_iterators + 1):
                batch_x, batch_y = mnist.train.next_batch(self.batch_size)
                sess.run(train_op, feed_dict={self.x: batch_x, self.y: batch_y})
                if step % 10 == 0 or step == self.max_iterators:
                    acc = sess.run(accuracy, feed_dict={self.x: test_images, self.y: test_labels})
                    print('Step %s, Accuracy %s' % (step, acc))
    