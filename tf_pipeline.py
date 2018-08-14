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
        self.y_out = None
        self.mean_loss = None
        self.train_step = None
        self.correct_prediction = None
        self.accuracy = None
        self.X = None
        self.Y = None
        self.is_training = None

        tf.reset_default_graph()

        self.X = tf.placeholder(tf.float32, [None, 32, 32, 3])
        self.y = tf.placeholder(tf.int64, [None])
        self.is_training = tf.placeholder(tf.bool)

        self.y_out = self.model_func(self.X, self.y, self.is_training)

        total_loss = tf.losses.softmax_cross_entropy(tf.one_hot(self.y,10), self.y_out)
        self.mean_loss = tf.reduce_mean(total_loss)
        optimizer = tf.train.RMSPropOptimizer(1e-3)

        # to support batch normalization
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_step = optimizer.minimize(self.mean_loss)

        self.correct_prediction = tf.equal(tf.argmax(self.y_out,1), self.y)
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

        self.tf_session = tf.Session()
        self.tf_session.run(tf.global_variables_initializer())

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.tf_session.close()

    def __compute_accuracy(self):
        pass
        
    def train_epoch(self, Xd, yd, batch_size=64):
        variables = [self.mean_loss, self.correct_prediction, self.accuracy, self.train_step]
        self.__run_step(variables, Xd, yd, True, batch_size)

    def validate(self, Xd, yd, batch_size=64):
        variables = [self.mean_loss, self.correct_prediction, self.accuracy]
        self.__run_step(variables, Xd, yd, False, batch_size)

    def __run_step(self, variables, Xd, yd, is_training, batch_size=64):
        correct = 0
        accuracies = []
        losses = []

        for indices in minibatch_indices(Xd.shape[0], batch_size):
            feed_dict = { self.X: Xd[indices, :],
                            self.y: yd[indices],
                            self.is_training: is_training }
            current_batch_size = indices.shape[0]
            loss, corr = self.tf_session.run(variables, feed_dict=feed_dict)[0:2]

            losses.append(loss * current_batch_size)
            correct += np.sum(corr)

        total_correct = correct/Xd.shape[0]
        accuracies.append(total_correct)
        total_loss = np.sum(losses)/Xd.shape[0]
        print("Overall loss = {0:.3g} and accuracy of {1:.3g}".format(total_loss,total_correct))
