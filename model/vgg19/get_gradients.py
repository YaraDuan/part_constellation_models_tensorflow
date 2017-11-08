import tensorflow as tf
import numpy as np
import scipy.io
import scipy.misc
import os
from pylab import *

IMAGE_W = 254
IMAGE_H = 254
ROOT = '/home/alala/Projects/part_constellation_models_tensorflow'
CONTENT_IMG = ROOT + '/images/plane.jpg'
OUTOUT_DIR = ROOT + '/results'
OUTPUT_IMG = 'results.png'
OUTPUT_MAT = 'results.mat'
VGG_MODEL = 'imagenet-vgg-verydeep-19.mat'
INI_NOISE_RATIO = 0.7
STYLE_STRENGTH = 500
ITERATION = 5000

LAYERS = ['pool5']

CONTENT_LAYERS =[('conv4_2',1.)]

MEAN_VALUES = np.array([123, 117, 104]).reshape((1, 1, 1, 3))


def build_net(ntype, nin, nwb=None):
    if ntype == 'conv':
        return tf.nn.relu(tf.nn.conv2d(nin, nwb[0], strides=[1, 1, 1, 1], padding='SAME') + nwb[1])
    elif ntype == 'pool':
        return tf.nn.avg_pool(nin, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1], padding='SAME')


def get_weight_bias(vgg_layers, i, ):
    weights = vgg_layers[i][0][0][0][0][0]
    weights = tf.constant(weights)
    bias = vgg_layers[i][0][0][0][0][1]
    bias = tf.constant(np.reshape(bias, (bias.size)))
    return weights, bias


def build_vgg19(path):
    net = {}
    vgg_rawnet = scipy.io.loadmat(path)
    vgg_layers = vgg_rawnet['layers'][0]
    net['input'] = tf.Variable(np.zeros((1, IMAGE_H, IMAGE_W, 3)).astype('float32'))
    net['conv1_1'] = build_net('conv', net['input'], get_weight_bias(vgg_layers, 0))
    net['conv1_2'] = build_net('conv', net['conv1_1'], get_weight_bias(vgg_layers, 2))
    net['pool1'] = build_net('pool', net['conv1_2'])
    net['conv2_1'] = build_net('conv', net['pool1'], get_weight_bias(vgg_layers, 5))
    net['conv2_2'] = build_net('conv', net['conv2_1'], get_weight_bias(vgg_layers, 7))
    net['pool2'] = build_net('pool', net['conv2_2'])
    net['conv3_1'] = build_net('conv', net['pool2'], get_weight_bias(vgg_layers, 10))
    net['conv3_2'] = build_net('conv', net['conv3_1'], get_weight_bias(vgg_layers, 12))
    net['conv3_3'] = build_net('conv', net['conv3_2'], get_weight_bias(vgg_layers, 14))
    net['conv3_4'] = build_net('conv', net['conv3_3'], get_weight_bias(vgg_layers, 16))
    net['pool3'] = build_net('pool', net['conv3_4'])
    net['conv4_1'] = build_net('conv', net['pool3'], get_weight_bias(vgg_layers, 19))
    net['conv4_2'] = build_net('conv', net['conv4_1'], get_weight_bias(vgg_layers, 21))
    net['conv4_3'] = build_net('conv', net['conv4_2'], get_weight_bias(vgg_layers, 23))
    net['conv4_4'] = build_net('conv', net['conv4_3'], get_weight_bias(vgg_layers, 25))
    net['pool4'] = build_net('pool', net['conv4_4'])
    net['conv5_1'] = build_net('conv', net['pool4'], get_weight_bias(vgg_layers, 28))
    net['conv5_2'] = build_net('conv', net['conv5_1'], get_weight_bias(vgg_layers, 30))
    net['conv5_3'] = build_net('conv', net['conv5_2'], get_weight_bias(vgg_layers, 32))
    net['conv5_4'] = build_net('conv', net['conv5_3'], get_weight_bias(vgg_layers, 34))
    net['pool5'] = build_net('pool', net['conv5_4'])
    return net


def build_content_loss(p, x):
    M = p.shape[1] * p.shape[2]
    N = p.shape[3]
    loss = (1. / (2 * N ** 0.5 * M ** 0.5)) * tf.reduce_sum(tf.pow((x - p), 2))
    return loss


def get_loss(p, x):
    print p
    print x
    label = np.ndarray([p.shape[0], p.shape[1], p.shape[2], p.shape[3]])
    for i in range(p.shape[0]):
        for j in range(p.shape[1]):
            for m in range(p.shape[2]):
                for n in range(p.shape[3]):
                    label[i][j][m][n] = p[i][j][m][n]
    label = label + 1
    loss = tf.reduce_sum(label - x)
    return loss


def get_label(sess,layer):
    p5 = sess.run(layer)
    label = np.ndarray([p5.shape[0], p5.shape[1], p5.shape[2], p5.shape[3]], dtype='float32')
    for i in range(p5.shape[0]):
        for j in range(p5.shape[1]):
            for m in range(p5.shape[2]):
                for n in range(p5.shape[3]):
                    label[i][j][m][n] = p5[i][j][m][n]
    label = label + 1
    return label


def read_image(path):
    image = scipy.misc.imread(path)
    image = scipy.misc.imresize(image, (IMAGE_H, IMAGE_W))
    image = image[np.newaxis, :, :, :]
    image = image - MEAN_VALUES
    return image


def write_image(path, image):
    """
    image = image + MEAN_VALUES
    image = image[0]
    image = np.clip(image, 0, 255).astype('uint8')
    """
    image = image.astype('uint8')

    scipy.misc.imsave(path, image)


def main():
    net = build_vgg19(VGG_MODEL)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    content_img = read_image(CONTENT_IMG)

    sess.run([net['input'].assign(content_img)])

    # get pool5

    pool5 = net['pool5']

    output = tf.gradients(pool5, net['input'])

    grad = sess.run(output)

    gradients_abs = np.abs(grad[0])

    result_img = np.sum(gradients_abs, axis=3)

    scipy.io.savemat(os.path.join(OUTOUT_DIR,OUTPUT_MAT), {'gradients':result_img[0]})





    result_img = result_img[0]
    # imshow(result_img)
    # print sess.run(loss)
    write_image(os.path.join(OUTOUT_DIR, OUTPUT_IMG), result_img)

    sess.close()


if __name__ == '__main__':
    main()