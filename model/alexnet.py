import tensorflow as tf
import numpy as np
import get_input_data as inputdata
from datetime import datetime
import os

FLAGS = tf.app.flags.FLAGS

class AlexNet(object):
    def __init__(self, x, keep_prob, num_classes, skip_layer,
                 weights_path='DEFAULT'):

        # Parse input arguments into class variables
        self.X = x
        self.NUM_CLASSES = num_classes
        self.KEEP_PROB = keep_prob
        self.SKIP_LAYER = skip_layer

        if weights_path == 'DEFAULT':
            self.WEIGHTS_PATH = 'bvlc_alexnet.npy'
        else:
            self.WEIGHTS_PATH = weights_path

        # Call the create function to build the computational graph of AlexNet
        self.create()

    def create(self):

        # 1st Layer: Conv (w ReLu) -> Pool -> Lrn
        conv1 = self.conv(self.X, 11, 11, 96, 4, 4, padding='VALID', name='conv1') # (55,55,96)
        pool1 = self.max_pool(conv1, 3, 3, 2, 2, padding='VALID', name='pool1')# (27,27,96)
        norm1 = self.lrn(pool1, 2, 2e-05, 0.75, name='norm1')#(27,27,96)

        # 2nd Layer: Conv (w ReLu) -> Pool -> Lrn with 2 groups
        conv2 = self.conv(norm1, 5, 5, 256, 1, 1, groups=2, name='conv2') #(27,27,256)
        pool2 = self.max_pool(conv2, 3, 3, 2, 2, padding='VALID', name='pool2')#(13,13,256)
        norm2 = self.lrn(pool2, 2, 2e-05, 0.75, name='norm2')#(13,13,256)

        # 3rd Layer: Conv (w ReLu)
        conv3 = self.conv(norm2, 3, 3, 384, 1, 1, name='conv3')#(13,13,384)

        # 4th Layer: Conv (w ReLu) splitted into two groups
        conv4 = self.conv(conv3, 3, 3, 384, 1, 1, groups=2, name='conv4')#(13,13,192) and (13,13,192)

        # 5th Layer: Conv (w ReLu) -> Pool splitted into two groups
        conv5 = self.conv(conv4, 3, 3, 256, 1, 1, groups=2, name='conv5')#(13,13,128)and(13,13,128)
        pool5 = self.max_pool(conv5, 3, 3, 2, 2, padding='VALID', name='pool5')#(6,6,256)

        # 6th Layer: Flatten -> FC (w ReLu) -> Dropout
        flattened = tf.reshape(pool5, [-1, 6 * 6 * 256])#turn into one line
        fc6 = self.fc(flattened, 6 * 6 * 256, 4096, name='fc6')
        dropout6 = self.dropout(fc6, self.KEEP_PROB)#(-1,4096)

        # 7th Layer: FC (w ReLu) -> Dropout
        fc7 = self.fc(dropout6, 4096, 4096, name='fc7')
        dropout7 = self.dropout(fc7, self.KEEP_PROB)#(4096)

        # 8th Layer: FC and return unscaled activations (for tf.nn.softmax_cross_entropy_with_logits)
        self.fc8 = self.fc(dropout7, 4096, self.NUM_CLASSES, relu=False, name='fc8')#(-1,2)

        return self.fc8

    def load_initial_weights(self, session):
        """
        As the weights from http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/ come
        as a dict of lists (e.g. weights['conv1'] is a list) and not as dict of
        dicts (e.g. weights['conv1'] is a dict with keys 'weights' & 'biases') we
        need a special load function
        """

        # Load the weights into memory
        weights_dict = np.load(self.WEIGHTS_PATH, encoding='bytes').item()

        # Loop over all layer names stored in the weights dict
        for op_name in weights_dict:

            # Check if the layer is one of the layers that should be reinitialized
            if op_name not in self.SKIP_LAYER:

                with tf.variable_scope(op_name, reuse=True):

                    # Loop over list of weights/biases and assign them to their corresponding tf variable
                    for data in weights_dict[op_name]:

                        # Biases
                        if len(data.shape) == 1:
                            var = tf.get_variable('biases', trainable=False)
                            session.run(var.assign(data))
                        # Weights
                        else:
                            var = tf.get_variable('weights', trainable=False)
                            session.run(var.assign(data))

    def conv(self, x, filter_height, filter_width, num_filters, stride_y, stride_x, name,
             padding='SAME', groups=1):
        """
        Adapted from: https://github.com/ethereon/caffe-tensorflow
        """
        # Get number of input channels
        input_channels = int(x.get_shape()[-1])

        # Create lambda function for the convolution
        convolve = lambda i, k: tf.nn.conv2d(i, k,
                                             strides=[1, stride_y, stride_x, 1],
                                             padding=padding)

        with tf.variable_scope(name) as scope:
            # Create tf variables for the weights and biases of the conv layer
            weights = tf.get_variable('weights',
                                      shape=[filter_height, filter_width, input_channels / groups, num_filters])
            biases = tf.get_variable('biases', shape=[num_filters])

            if groups == 1:
                conv = convolve(x, weights)

            # In the cases of multiple groups, split inputs & weights and
            else:
                # Split input and weights and convolve them separately
                input_groups = tf.split(axis=3, num_or_size_splits=groups, value=x)
                weight_groups = tf.split(axis=3, num_or_size_splits=groups, value=weights)
                output_groups = [convolve(i, k) for i, k in zip(input_groups, weight_groups)]

                # Concat the convolved output together again
                conv = tf.concat(axis=3, values=output_groups)

            # Add biases
            bias = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list())

            # Apply relu function
            relu = tf.nn.relu(bias, name=scope.name)

            return relu

    def max_pool(self, x, filter_height, filter_width, stride_y, stride_x, name, padding='SAME'):
        return tf.nn.max_pool(x, ksize=[1, filter_height, filter_width, 1],
                              strides=[1, stride_y, stride_x, 1],
                              padding=padding, name=name)

    def lrn(self, x, radius, alpha, beta, name, bias=1.0):
        return tf.nn.local_response_normalization(x, depth_radius=radius, alpha=alpha,
                                                  beta=beta, bias=bias, name=name)

    def dropout(self, x, keep_prob):
        return tf.nn.dropout(x, keep_prob)

    def fc(self, x, num_in, num_out, name, relu=True):
        with tf.variable_scope(name) as scope:

            # Create tf variables for the weights and biases
            weights = tf.get_variable('weights', shape=[num_in, num_out], trainable=True)
            biases = tf.get_variable('biases', [num_out], trainable=True)

            # Matrix multiply weights and inputs and add bias
            act = tf.nn.xw_plus_b(x, weights, biases, name=scope.name)

            if relu == True:
                # Apply ReLu non linearity
                relu = tf.nn.relu(act)
                return relu
            else:
                return act


def train():

    '''
    # params of model
    tf.app.flags.DEFINE_integer('batch_size', 128,
                                """Number of images to process in a batch.""")

    tf.app.flags.DEFINE_string('train_file', 'D:\\train_file.txt',
                               """Path to the  train data directory.""")
    tf.app.flags.DEFINE_string('val_file', 'D:\\val_file.txt',
                               """Path to the  val data directory.""")
    # tf.app.flags.DEFINE_integer('max_steps',10000,"""Number of batches to run.""")
    '''

    # global var

    keep_p = 0.5
    IMAGE_SIZE = 227
    NUM_CLASSES = 23
    BATCH_SIZE = 64
    lr = 0.01
    NUM_EPOCH = 10
    train_layers = ['fc8', 'fc7']

    # How often we want to write the tf.summary data to disk
    display_step = 1

    filewriter_path = '../../alexnet_logs'
    checkpoint_path = '../../alexnet_logs'

    # TF placeholder for graph input and output
    x = tf.placeholder(tf.float32, [BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3])
    y = tf.placeholder(tf.float32, [None, NUM_CLASSES])
    keep_prob = tf.placeholder(tf.float32)

    # init the model
    model = AlexNet(x, keep_prob, NUM_CLASSES, train_layers)

    score = model.fc8

    #var_list = [v for v in tf.trainable_variables() if v.name.split('/')[0] in train_layers]

    # loss
    with tf.name_scope('cross_ent'):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=score, labels=y))


    # train op
    with tf.name_scope('train'):
        #gradients = tf.gradients(loss, var_list)
        #gradients = list(zip(gradients, var_list))

        # define the loss function and learnning rate
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=y))
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
        train_op = optimizer

        #optimizer = tf.train.GradientDescentOptimizer(lr)
        #train_op = optimizer.apply_gradients(grads_and_vars=gradients)

    '''
    # add gradients to summary
    for gradient, var in gradients:
        tf.summary.histogram(var.name + '/gradients', gradient)
    # add the variables we train to the summary
    for var in var_list:
        tf.summary.histogram(var.name, var)
    '''

    # add the loss to summary
    tf.summary.scalar('cross_entropy', loss)

    # evaluation op: accuarcy
    with tf.name_scope('accuracy')as scope:
        correct_pred = tf.equal(tf.argmax(score, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Add accuracy into summary
    tf.summary.scalar('accuracy', accuracy)

    merged = tf.summary.merge_all()

    # init
    writer = tf.summary.FileWriter(filewriter_path)
    saver = tf.train.Saver()

    # load data
    #train_generator = ImageDataGenerator(FLAGS.train_file, horizontal_flip=True, shuffle=True)
    #val_generator = ImageDataGenerator(FLAGS.val_file, shuffle=False)
    # train_image,train_label=inputs(FLAGS.datadir,FLAGS.batch_size)
    # test_image,test_label=inputs(FLAGS.datadir,FLAGS.batch_size,shuffle=False)
    image_list, label_list = inputdata.get_data_list('../../data/plane23/train')

    train_batches_per_epoch = np.floor(image_list.size / FLAGS.batch_size).astype(np.int16)
    #val_batches_per_epoch = np.floor(val_generator.size / FLAGS.batch_size).astype(np.int16)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        writer.add_graph(sess.graph)
        model.load_initial_weights(sess)
        print("{} start training..".format(datetime.now()))
        print("{} Open Tensorboard at --logdir {}".format(datetime.now(),
                                                          filewriter_path))

        for epoch in range(NUM_EPOCH):
            print("{} Epoch number: {}".format(datetime.now(), epoch + 1))
            step = 1
            while step < train_batches_per_epoch:
                # get a batch
                batch_xs, batch_ys = inputdata.get_batch(image_list,label_list,227,227,64,256)

                # train
                sess.run(train_op, feed_dict={x: batch_xs, y: batch_ys, keep_prob: keep_p})

                # generate summary
                if step % display_step == 0:
                    s = sess.run(merged, feed_dict={x: batch_xs,
                                                    y: batch_ys,
                                                    keep_prob: 1.})
                    writer.add_summary(s, epoch * train_batches_per_epoch + step)
                step += 1

            print 'finished train'

            '''
            # validate the model on the entire validation set
            print('{} Start validation'.format(datetime.now()))
            test_acc = 0
            test_count = 0
            for _ in range(val_batches_per_epoch):
                batch_tx, batch_ty = val_generator.next_batch(FLAGS.batch_size)
                acc = sess.run(accuracy, feed_dict={x: batch_tx, y: batch_ty, keep_prob: 1.})
                test_acc += acc
            test_acc /= test_count
            print('Validation Accuracy =%f' % test_acc)

            # reset the file pointer of the image data generator
            val_generator.reset_pointer()
            train_generator.reset_pointer()
            print('{}Saving checkpoint of model..'.format(datetime.now()))
            '''

            # save checkpoint of the model
            checkpoint_name = os.path.join(checkpoint_path, 'model_epoch' + str(epoch + 1) + '.ckpt')
            save_path = saver.save(sess, checkpoint_name)
            print("{} Model checkpoint saved at {}".format(datetime.now(), checkpoint_name))


if __name__ == '__main__':
    train()