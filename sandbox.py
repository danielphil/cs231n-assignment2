import tf_pipeline

import numpy as np
import tensorflow as tf

from cs231n.data_utils import load_CIFAR10

def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=10000):
    """
    Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
    it for the two-layer neural net classifier. These are the same steps as
    we used for the SVM, but condensed to a single function.  
    """
    # Load the raw CIFAR-10 data
    cifar10_dir = 'cs231n/datasets/cifar-10-batches-py'
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

    # Subsample the data
    mask = range(num_training, num_training + num_validation)
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = range(num_training)
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = range(num_test)
    X_test = X_test[mask]
    y_test = y_test[mask]

    # Normalize the data: subtract the mean image
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image

    return X_train, y_train, X_val, y_val, X_test, y_test


# Invoke the above function to get our data.
X_train, y_train, X_val, y_val, X_test, y_test = get_CIFAR10_data()
print('Train data shape: ', X_train.shape)
print('Train labels shape: ', y_train.shape)
print('Validation data shape: ', X_val.shape)
print('Validation labels shape: ', y_val.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', y_test.shape)

def sandbox_model(X,y,is_training, filter_size=2):  
    # 32 layer conv2d with 3x3 filters
    h1 = tf.layers.conv2d(X, 32, [filter_size, filter_size], strides=(1, 1), activation=tf.nn.relu)
    
    # batch normalization
    bn = tf.layers.batch_normalization(h1, axis=3, training=is_training)

    # max pooling 2x2 with stride 2
    # max_pool shape (?, 15, 15, 32)
    max_pool = tf.layers.max_pooling2d(bn, [2, 2], [2, 2])

    # affine layer with 1024 output units and relu
    inputs = max_pool.shape[1] * max_pool.shape[2] * max_pool.shape[3]
    max_pool_flat = tf.reshape(max_pool,[-1,inputs])
    h2 = tf.layers.dense(max_pool_flat, 1024, activation=tf.nn.relu)
    
    # affine layer 2 with 10 outputs  
    y_out = tf.layers.dense(h2, 10, activation=None)
    
    print("I finished!")
    return y_out

pipeline = tf_pipeline.TfPipeline(sandbox_model)

#with tf_pipeline.TfPipeline(sandbox_model) as pipeline:
#    pipeline.run(X_train, y_train, X_val, y_val)