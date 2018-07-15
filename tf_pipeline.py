import tensorflow as tf
import numpy as np
import math

def minibatch_indices(N, batch_size):
    batches = int(math.ceil(N/batch_size))

    indices = np.arange(N)
    np.random.shuffle(indices)

    for current_batch in range(batches):
        start_index = current_batch * batch_size
        yield indices[start_index:start_index+batch_size]

class TfPipeline:
    def __init__(self, model_func):
        self.model_func = model_func

    def __build_graph(self):
        tf.reset_default_graph()

        X = tf.placeholder(tf.float32, [None, 32, 32, 3])
        y = tf.placeholder(tf.int64, [None])
        is_training = tf.placeholder(tf.bool)

        self.y_out = self.model_func(X, y, is_training)

        total_loss = tf.losses.softmax_cross_entropy(tf.one_hot(y,10), y_out)
        mean_loss = tf.reduce_mean(total_loss)
        optimizer = tf.train.RMSPropOptimizer(1e-3)

        # to support batch normalization
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_step = optimizer.minimize(mean_loss)

    def __compute_accuracy(self):
        correct_prediction = tf.equal(tf.argmax(self.y_out,1), y)
        return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def __run_model(self):
        accuracy = self.__compute_accuracy()


    def run(self, X_train, y_train, X_val, y_val):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())



    # run_model_short(sess,y_out,mean_loss,X_train,y_train,10,64,100,train_step,True)