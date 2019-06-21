import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
from utils import batch
import numpy as np
import logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
    logging.warning("GPU device not found. Without a GPU the execution of the program will be very slow.")
else:
    print('Found GPU at: {}'.format(device_name))



def matrix_add(ma, mb):
    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, [len(ma), len(ma[0])])
    Y = tf.placeholder(tf.float32, [len(mb), len(mb[0])])

    m = tf.add(X, Y)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    x = sess.run(m, {X: ma, Y: mb})
    sess.close()
    tf.reset_default_graph()
    return x


def matrix_dot(ma, mb):
    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, [len(ma), len(ma[0])])
    Y = tf.placeholder(tf.float32, [len(mb), len(mb[0])])

    m = tf.matmul(X, tf.transpose(Y))

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    x = sess.run(m, {X: ma, Y: mb})
    sess.close()
    tf.reset_default_graph()
    return x



def matrix_dot_batches(ma,mb,batch_size=10000):
    x = None

    for i_batch, mbatch in enumerate(batch(mb, batch_size)):
        if x is None:
            x = matrix_dot(ma, mbatch)
        else:
            x = np.concatenate((x, matrix_dot(ma, mbatch)), axis=1)

    return x


def k_top(ma, k):
    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, [len(ma), len(ma[0])])

    _, indexes = tf.nn.top_k(ma, k)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    x = sess.run(indexes, {X: ma})
    sess.close()
    tf.reset_default_graph()
    return x


def cosine_knn(ma, mb, k):
    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, [len(ma), len(ma[0])])
    Y = tf.placeholder(tf.float32, [len(mb), len(mb[0])])

    m = tf.matmul(X, tf.transpose(Y))
    _, indexes = tf.nn.top_k(m, k)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    x = sess.run(indexes, {X: ma, Y: mb})
    sess.close()
    tf.reset_default_graph()
    return x


def cosine_knn_batches(ma, mb, k, batch_size=10000):

    #mDot = matrix_dot_batches(ma, mb, batch_size=batch_size)

    x = None

    for i_batch, mbatch in enumerate(batch(mb, batch_size)):
        if x is None:
            x = matrix_dot(ma, mbatch)
        else:
            x = np.concatenate((x, matrix_dot(ma, mbatch)), axis=1)

    top = k_top(x, k)
    return top


def matrix_analogy(ma,mb,mc,mM):
    tf.reset_default_graph()
    a = tf.placeholder(tf.float32, [len(ma), len(ma[0])])
    b = tf.placeholder(tf.float32, [len(mb), len(mb[0])])
    c = tf.placeholder(tf.float32, [len(mc), len(mc[0])])
    M = tf.placeholder(tf.float32, [len(mM), len(mM[0])])

    ag = tf.add(tf.subtract(c, a), b)
    nn = tf.matmul(ag, tf.transpose(M))

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    x = sess.run(nn, {a: ma, b: mb, c: mc, M:mM})
    sess.close()
    tf.reset_default_graph()

    return x

