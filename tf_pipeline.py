import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt
import random
import json

def random_search_params(options, limit = 1000):
    max_limit = 1
    for key in options:
        max_limit *= len(options[key])

    limit = min(max_limit, limit)
    
    previous_hashes = set()
    while len(previous_hashes) < limit:
        output_options = {}
        for key in options:
            index = random.randrange(len(options[key]))
            output_options[key] = options[key][index]
        
        # Don't generate duplicates
        options_hash = hash(json.dumps(output_options, sort_keys=True))
        if options_hash in previous_hashes:
            continue
        previous_hashes.add(options_hash)

        yield output_options

def minibatch_indices(N, batch_size):
    batches = int(math.ceil(N/batch_size))

    indices = np.arange(N)
    np.random.shuffle(indices)

    for current_batch in range(batches):
        start_index = current_batch * batch_size
        yield indices[start_index:start_index+batch_size]

def train_and_plot(sandbox_model, X_train, y_train, X_val, y_val, epochs=20):
    with TfPipeline(sandbox_model) as pipeline:
        last_accuracy = 0
        train_accuracies = []
        val_accuracies = []

        for epoch in range(epochs):
            print('.', end='')
            #print("Training Epoch {0}:".format(epoch + 1), end='')
            train_loss, train_accuracy = pipeline.train_epoch(X_train, y_train)
            #print("Overall loss = {0:.3g} and accuracy of {1:.3g}".format(train_loss,train_accuracy))
            train_accuracies.append(train_accuracy)

            #print("Validating Epoch {0}:".format(epoch + 1), end='')
            val_loss, val_accuracy = pipeline.validate(X_val, y_val)
            #print("Overall loss = {0:.3g} and accuracy of {1:.3g}".format(val_loss,val_accuracy))
            val_accuracies.append(val_accuracy)

            if epoch % 5 == 0:
                if val_accuracy < last_accuracy:
                    # we're getting worse, stop early!
                    print("Stopping early!")
                    break

                last_accuracy = val_accuracy

        plt.plot(train_accuracies, 'b-', label="Training")
        plt.plot(val_accuracies, 'r-', label="Validation")
        plt.grid(True)
        plt.title('Accuracies')
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.legend(loc='best')
        plt.show()

        print()
        print("Training accuracy of {0:.3g}".format(train_accuracies[-1]))
        print("Validation accuracy of {0:.3g}".format(val_accuracies[-1]))

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
        return self.__run_step(variables, Xd, yd, True, batch_size)

    def validate(self, Xd, yd, batch_size=64):
        variables = [self.mean_loss, self.correct_prediction, self.accuracy]
        return self.__run_step(variables, Xd, yd, False, batch_size)

    def __run_step(self, variables, Xd, yd, is_training, batch_size=64):
        correct = 0
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
        total_loss = np.sum(losses)/Xd.shape[0]
        return loss, total_correct