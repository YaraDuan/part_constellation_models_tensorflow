# coding: UTF-8

import tensorflow as tf
import numpy as np
import os
import get_input_data as input
from datetime import datetime
import matplotlib.pyplot as plt


def maxPoolLayer(x, kHeight, kWidth, strideX, strideY, name, padding = "SAME"):
    """max-pooling"""
    return tf.nn.max_pool(x, ksize = [1, kHeight, kWidth, 1],
                          strides = [1, strideX, strideY, 1], padding = padding, name = name)


def dropout(x, keepPro, name = None):
    """dropout"""
    return tf.nn.dropout(x, keepPro, name)


def fcLayer(x, inputD, outputD, reluFlag, name):
    """fully-connect"""
    with tf.variable_scope(name) as scope:
        w = tf.get_variable("w", shape = [inputD, outputD], dtype = "float")
        b = tf.get_variable("b", [outputD], dtype = "float")
        out = tf.nn.xw_plus_b(x, w, b, name = scope.name)
        if reluFlag:
            return tf.nn.relu(out)
        else:
            return out


def convLayer(x, kHeight, kWidth, strideX, strideY,
              featureNum, name, padding = "SAME"):
    """convlutional"""
    channel = int(x.get_shape()[-1])
    with tf.variable_scope(name) as scope:
        w = tf.get_variable("w", shape=[kHeight, kWidth, channel, featureNum])
        b = tf.get_variable("b", shape=[featureNum])
        featureMap = tf.nn.conv2d(x, w, strides=[1, strideY, strideX, 1], padding=padding)
        out = tf.nn.bias_add(featureMap, b)

        return tf.nn.relu(tf.reshape(out, featureMap.get_shape().as_list()), name=scope.name)


class VGG19(object):
    """VGG model"""
    def __init__(self, x, keepPro, classNum, skip, modelPath = "vgg19.npy"):
        self.X = x
        self.KEEPPRO = keepPro
        self.CLASSNUM = classNum
        self.SKIP = skip
        self.MODELPATH = modelPath
        #build CNN
        self.build_vgg19()

    def build_vgg19(self):
        """build model"""
        net = {}
        net['conv1_1'] = convLayer(self.X, 3, 3, 1, 1, 64, "conv1_1")
        net['conv1_2'] = convLayer(net['conv1_1'], 3, 3, 1, 1, 64, "conv1_2")
        net['pool1'] = maxPoolLayer(net['conv1_2'], 2, 2, 2, 2, "pool1")

        net['conv2_1'] = convLayer(net['pool1'], 3, 3, 1, 1, 128, "conv2_1")
        net['conv2_2'] = convLayer(net['conv2_1'], 3, 3, 1, 1, 128, "conv2_2")
        net['pool2'] = maxPoolLayer(net['conv2_2'], 2, 2, 2, 2, "pool2")

        net['conv3_1'] = convLayer(net['pool2'], 3, 3, 1, 1, 256, "conv3_1")
        net['conv3_2'] = convLayer(net['conv3_1'], 3, 3, 1, 1, 256, "conv3_2")
        net['conv3_3'] = convLayer(net['conv3_2'], 3, 3, 1, 1, 256, "conv3_3")
        net['conv3_4'] = convLayer(net['conv3_3'], 3, 3, 1, 1, 256, "conv3_4")
        net['pool3'] = maxPoolLayer(net['conv3_4'], 2, 2, 2, 2, "pool3")

        net['conv4_1'] = convLayer(net['pool3'], 3, 3, 1, 1, 512, "conv4_1")
        net['conv4_2'] = convLayer(net['conv4_1'], 3, 3, 1, 1, 512, "conv4_2")
        net['conv4_3'] = convLayer(net['conv4_2'], 3, 3, 1, 1, 512, "conv4_3")
        net['conv4_4'] = convLayer(net['conv4_3'], 3, 3, 1, 1, 512, "conv4_4")
        net['pool4'] = maxPoolLayer(net['conv4_4'], 2, 2, 2, 2, "pool4")

        net['conv5_1'] = convLayer(net['pool4'], 3, 3, 1, 1, 512, "conv5_1")
        net['conv5_2'] = convLayer(net['conv5_1'], 3, 3, 1, 1, 512, "conv5_2")
        net['conv5_3'] = convLayer(net['conv5_2'], 3, 3, 1, 1, 512, "conv5_3")
        net['conv5_4'] = convLayer(net['conv5_3'], 3, 3, 1, 1, 512, "conv5_4")
        net['pool5'] = maxPoolLayer(net['conv5_4'], 2, 2, 2, 2, "pool5")

        fcIn = tf.reshape(net['pool5'], [-1, int(net['pool5'].shape[1]) * int(net['pool5'].shape[2]) * 512])
        net['fc6'] = fcLayer(fcIn, int(net['pool5'].shape[1]) * int(net['pool5'].shape[2]) * 512, 4096, True, "fc6")
        net['dropout1'] = dropout(net['fc6'], self.KEEPPRO)

        net['fc7'] = fcLayer(net['dropout1'], 4096, 4096, True, "fc7")
        net['dropout2'] = dropout(net['fc7'], self.KEEPPRO)

        net['fc8'] = fcLayer(net['dropout2'], 4096, self.CLASSNUM, True, "fc8")

        self.vgg19_net = net

    def loadModel(self, sess):
        """load model"""
        wDict = np.load(self.MODELPATH, encoding = "bytes").item()
        #for layers in model
        for name in wDict:
            if name not in self.SKIP:
                with tf.variable_scope(name, reuse = True):
                    for p in wDict[name]:
                        if len(p.shape) == 1:
                            #bias
                            sess.run(tf.get_variable('b', trainable = False).assign(p))
                        else:
                            #weights
                            sess.run(tf.get_variable('w', trainable = False).assign(p))

    # Calculate loss
    def get_loss(self, predicts, labels):

        #predicts = tf.nn.softmax(predicts)

        #labels = tf.one_hot(labels, self.vgg19_net['fc8'].get_shape().as_list()[1])

        loss = tf.nn.softmax_cross_entropy_with_logits(logits=predicts, labels=labels)

        cost = tf.reduce_mean(loss)

        tf.summary.scalar('loss', cost)

        return cost

    # Get gradients
    def optimer(self, loss, lr=0.001):

        train_optimizer = tf.train.GradientDescentOptimizer(lr).minimize(loss)

        return train_optimizer

    # Calculate accuracy
    def accuracy(self, predicts, labels):
        """
        acc = tf.nn.in_top_k(predicts, labels, 1)

        acc = tf.cast(acc, tf.float32)

        acc = tf.reduce_mean(acc)
        """

        #labels = tf.one_hot(labels, self.vgg19_net['fc8'].get_shape().as_list()[1])

        correct_pred = tf.equal(tf.argmax(predicts, 1), tf.argmax(labels, 1))
        #acc=correct_pred

        acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        return acc


