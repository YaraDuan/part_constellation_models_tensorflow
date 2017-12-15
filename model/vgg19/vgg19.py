# coding: UTF-8

import tensorflow as tf
import numpy as np
import scipy.io
import os
import get_input_data as input
from datetime import datetime
import matplotlib.pyplot as plt


def maxPoolLayer(x, kHeight, kWidth, strideX, strideY, name, padding="SAME"):
    """max-pooling"""
    return tf.nn.max_pool(x, ksize=[1, kHeight, kWidth, 1],
                          strides=[1, strideX, strideY, 1], padding=padding, name=name)


def dropout(x, keepPro, name=None):
    """dropout"""
    return tf.nn.dropout(x, keepPro, name)


def fcLayer(x, inputD, outputD, reluFlag, name):
    """fully-connect"""
    with tf.variable_scope(name) as scope:
        w = tf.get_variable("w", shape=[inputD, outputD], dtype="float")
        b = tf.get_variable("b", [outputD], dtype="float")
        """
        if w_and_b[0].shape[-1] > outputD:
            w = w_and_b[0][:, :, :, 0:outputD]
            b = w_and_b[1][0:outputD]
            w = tf.reshape(w, [inputD, outputD])
        else:
            w = tf.reshape(w_and_b[0], [inputD, outputD])
            b = w_and_b[1]
        """
        out = tf.nn.xw_plus_b(x, w, b, name=scope.name)

        #tf.summary.histogram('w', w)
        #tf.summary.histogram('b', b)
        if reluFlag:
            #tf.summary.histogram('output', tf.nn.relu(out))
            return tf.nn.relu(out)
        else:
            #tf.summary.histogram('output', out)
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

        #tf.summary.histogram('w', w)
        #tf.summary.histogram('b', b)
        #tf.summary.histogram('output', tf.nn.relu(tf.reshape(out, featureMap.get_shape().as_list()), name=scope.name))
        return tf.nn.relu(tf.reshape(out, featureMap.get_shape().as_list()), name=scope.name)


def softmaxLayer(x, name):
    with tf.variable_scope(name) as scope:
        pred = tf.nn.softmax(x, name=scope.name)
        #tf.summary.histogram('softmax', pred)
        return pred


def get_weight_bias(vgg_layers, i):
    weights = vgg_layers[i][0][0][0][0][0]
    #weights = tf.constant(weights)
    bias = vgg_layers[i][0][0][0][0][1]
    bias = np.reshape(bias, (bias.size))
    #bias = tf.constant(np.reshape(bias, (bias.size)))
    return weights, bias


class VGG19(object):
    """VGG model"""
    def __init__(self, x, keepPro, classNum, skip):
        self.X = x
        self.KEEPPRO = keepPro
        self.CLASSNUM = classNum
        self.SKIP = skip
        #self.MODELPATH = modelPath
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

        net['softmax'] = softmaxLayer(net['fc8'], "softmax")

        self.vgg19_net = net

    def loadModel(self, sess=None, model_mode=None, modelpath=None):
        """load model"""
        if model_mode == 'MATLAB':
            vgg_rawnet = scipy.io.loadmat(modelpath)
            vgg_layers = vgg_rawnet['layers'][0]

            layer_index_in_pretrained_model = np.array([0, 2, 5, 7, 10, 12, 14, 16, 19, 21, 23, 25, 28, 30, 32, 34, 37, 39, 41])
            layers = []
            keys = self.vgg19_net.keys()
            for k in keys:
                if k.find('conv') >= 0 or k.find('fc') >= 0:
                    layers.append(k)
            layers = np.array(sorted(layers))
            name_scope = np.transpose(np.vstack((layers, layer_index_in_pretrained_model)), [1, 0])

            for layer in name_scope:
                with tf.variable_scope(layer[0], reuse=True):
                    w, b = get_weight_bias(vgg_layers, int(layer[1]))

                    if layer[0].find('fc') >= 0:
                        # train the variables of fc layer
                        tensor_shape = tf.get_variable('w', trainable=False).get_shape()
                        if w.shape[-1] > tensor_shape[-1]:
                            w = w[:, :, :, 0:tensor_shape[-1]]
                            b = b[0:tensor_shape[-1]]
                        w = np.reshape(w, tensor_shape)
                        sess.run(tf.get_variable('w', trainable=True).assign(w))
                        sess.run(tf.get_variable('b', trainable=True).assign(b))
                    elif layer[0].find('conv') >= 0:
                        # If the layer is conv, do not train.
                        sess.run(tf.get_variable('w', trainable=False).assign(w))
                        sess.run(tf.get_variable('b', trainable=False).assign(b))

        else:
            # wDict = np.load(modelpath, encoding="bytes").item()
            layers = []
            keys = self.vgg19_net.keys()
            for k in keys:
                if k.find('conv') >= 0 or k.find('fc') >= 0:
                    layers.append(k)
            wDict = np.array(sorted(layers))
            # for layers in model
            for name in wDict:
                if name not in self.SKIP:
                    with tf.variable_scope(name, reuse=True):
                        sess.run(tf.get_variable('b', trainable=False))
                        sess.run(tf.get_variable('w', trainable=False))
                        """
                        for p in wDict[name]:
                            if len(p.shape) == 1:
                                # bias
                                sess.run(tf.get_variable('b', trainable=False).assign(p))
                            else:
                                # weights
                                sess.run(tf.get_variable('w', trainable=False).assign(p))
                        """
    # Calculate loss
    def get_loss(self, predicts, labels):

        #predicts = tf.nn.softmax(predicts)

        labels = tf.one_hot(labels, self.vgg19_net['fc8'].get_shape().as_list()[1])
        """
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=predicts, labels=labels)

        cost = tf.reduce_mean(loss)

        reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

        reg_constant = 0.0005

        cost = cost + reg_constant * sum(reg_loss)

        # cost = -tf.reduce_mean(labels*tf.log(tf.clip_by_value(predicts, 1e-10, 1.0)))
        tf.summary.scalar('loss', cost)
        """
        labels = tf.cast(labels, dtype=tf.float32)
        cost = tf.reduce_mean(-tf.reduce_sum(labels*tf.log(predicts), reduction_indices=1))
        tf.summary.scalar('loss', cost)

        return cost

    # Get gradients
    def optimer(self, loss, lr=0.001):

        train_optimizer = tf.train.GradientDescentOptimizer(lr).minimize(loss)

        return train_optimizer

    # Calculate accuracy
    def accuracy_top_1(self, predicts, labels):

        acc = tf.nn.in_top_k(predicts, labels, 1)

        acc = tf.cast(acc, tf.float32)

        acc = tf.reduce_mean(acc)


        #labels = tf.one_hot(labels, self.vgg19_net['fc8'].get_shape().as_list()[1])

        #correct_pred = tf.equal(tf.argmax(predicts, 1), tf.argmax(labels, 1))
        #acc=correct_pred

        #acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        tf.summary.scalar('accuracy_top_1', acc)

        return acc

    def accuracy_top_5(self, predicts, labels):

        acc = tf.nn.in_top_k(predicts, labels, 5)
        acc = tf.cast(acc, tf.float32)
        acc = tf.reduce_mean(acc)
        tf.summary.scalar('accracy_top_5', acc)

        return acc


