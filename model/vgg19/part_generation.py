import tensorflow as tf
import numpy as np
import scipy.io
import scipy.misc
from scipy import ndimage
import os
import cv2
from pylab import *
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

IMAGE_W = 800
IMAGE_H = 600
ROOT = '/home/alala/Projects/part_constellation_models_tensorflow'
CONTENT_IMG = ROOT + '/images/Taipei101.jpg'
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


def get_label(channel_id,layer):
    label = np.ndarray([layer.shape[0], layer.shape[1], layer.shape[2], layer.shape[3]], dtype='float32')
    for b in range(layer.shape[0]):
        for w in range(layer.shape[1]):
            for h in range(layer.shape[2]):
                for c in range(layer.shape[3]):
                    if c == channel_id:
                        label[b][w][h][c] = 1
                    else:
                        label[b][w][h][c] = 0
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


def get_gradients():

    gmaps = []

    batch_size = 64

    for batch_id in range(512 / batch_size):

        net = build_vgg19(VGG_MODEL)
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        content_img = read_image(CONTENT_IMG)

        sess.run([net['input'].assign(content_img)])

        # get pool5
        pool5 = net['pool5']

        for i in range(batch_size):
            init = get_label((batch_size * batch_id + i), pool5)

            # get gradients
            output = tf.gradients(pool5 * init, net['input'])
            grad = sess.run(output)

            gmaps.append(grad[0][0])

            print batch_size * batch_id + i

        sess.close()

    #gmaps = np.random.randint(-5,5,(512,IMAGE_W,IMAGE_W,3))

    return gmaps


def fitGMMToGradient(gmap):

    if len(np.where(gmap[:] != 0)):

        gmap = ndimage.gaussian_filter(gmap, 3)

        ys, xs = np.where(np.max(gmap[:]) == gmap)

        est_x = xs[len(xs)-1]
        est_y = ys[len(ys)-1]
    else:
        est_x = -1
        est_y = -1

    return est_x, est_y


def part_generation():

    imagelist = 1

    part_locs = []
    """
    for i in range(imagelist):

        g = get_gradients()

        for p in range(512):

            gmap = np.squeeze(np.sum(np.abs(g[p]), 2))

            if np.sum(np.isnan(gmap)) > 0 or len(np.where(gmap[:] != 0)[0]) < 1:
                continue

            est_x, est_y = fitGMMToGradient(gmap)

            if est_x == -1 and est_y == -1:
                part_locs.append([i, p, est_x, est_y, 0])
            else:
                part_locs.append([i, p, est_x, est_y, 1])

    scipy.io.savemat(os.path.join(OUTOUT_DIR, 'part_locs.mat'), {'part_locs': part_locs})
    """
    part_locs = scipy.io.loadmat(os.path.join(OUTOUT_DIR, 'part_locs.mat'))

    part_locs = part_locs['part_locs']

    # draw the dots
    im = plt.imread(CONTENT_IMG)
    plt.figure(), plt.imshow(im)

    for i in range(len(part_locs)):

        info = part_locs[i]

        x = info[2]

        y = info[3]

        plt.plot(x, y, 'rx')

    plt.show()
    plt.imsave(os.path.join(OUTOUT_DIR, OUTPUT_IMG), im)

    return part_locs


#if __name__ == '__main__':
#    main()