import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import get_input_data as inputdata

crop_w = crop_h = 227
batch_size = 64
capacity = 256

# load data
#mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
train_dir = './data/plane23/train'
train_data, train_label = inputdata.get_data_list(train_dir)
train_data_batch, train_label_batch = inputdata.get_batch(train_data,train_label,crop_w,crop_h,batch_size,capacity)

# define some paras
learning_rate = 0.001
training_iters = 200000
batch_size = 64
display_step = 20

n_input = 51529 # dimension of input data
n_classes = 23 # dimension of label
dropout = 0.8 # probablity of dropout

# placeholder
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32)


def conv2d(name, l_input, w, b):
    return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(l_input, w, strides=[1, 1, 1, 1], padding='SAME'),b), name=name)

def max_pool(name, l_input, k):
    return tf.nn.max_pool(l_input, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME', name=name)

def norm(name, l_input, lsize=4):
    return tf.nn.lrn(l_input, lsize, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name=name)

# define the net
def alex_net(_X, _weights, _biases, _dropout):
    # turn vector to matrix
    _X = tf.reshape(_X, shape=[-1, 227, 227, 3])

    # conv
    conv1 = conv2d('conv1', _X, _weights['wc1'], _biases['bc1'])
    # pool
    pool1 = max_pool('pool1', conv1, k=2)
    # norm
    norm1 = norm('norm1', pool1, lsize=4)
    # Dropout
    norm1 = tf.nn.dropout(norm1, _dropout)

    # conv
    conv2 = conv2d('conv2', norm1, _weights['wc2'], _biases['bc2'])
    # pool
    pool2 = max_pool('pool2', conv2, k=2)
    # norm
    norm2 = norm('norm2', pool2, lsize=4)
    # Dropout
    norm2 = tf.nn.dropout(norm2, _dropout)

    # conv
    conv3 = conv2d('conv3', norm2, _weights['wc3'], _biases['bc3'])
    # pool
    pool3 = max_pool('pool3', conv3, k=2)
    # norm
    norm3 = norm('norm3', pool3, lsize=4)
    # Dropout
    norm3 = tf.nn.dropout(norm3, _dropout)

    # fully connect layer, turn feature map to vector firstly.
    dense1 = tf.reshape(norm3, [-1, _weights['wd1'].get_shape().as_list()[0]])
    dense1 = tf.nn.relu(tf.matmul(dense1, _weights['wd1']) + _biases['bd1'], name='fc1')
    # fully connect layer
    dense2 = tf.nn.relu(tf.matmul(dense1, _weights['wd2']) + _biases['bd2'], name='fc2') # Relu activation

    # output
    out = tf.matmul(dense2, _weights['out']) + _biases['out']
    return out

# store the params
weights = {
    'wc1': tf.Variable(tf.random_normal([3, 3, 1, 64])),
    'wc2': tf.Variable(tf.random_normal([3, 3, 64, 128])),
    'wc3': tf.Variable(tf.random_normal([3, 3, 128, 256])),
    'wd1': tf.Variable(tf.random_normal([4*4*256, 1024])),
    'wd2': tf.Variable(tf.random_normal([1024, 1024])),
    'out': tf.Variable(tf.random_normal([1024, 23]))
}
biases = {
    'bc1': tf.Variable(tf.random_normal([64])),
    'bc2': tf.Variable(tf.random_normal([128])),
    'bc3': tf.Variable(tf.random_normal([256])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'bd2': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# build the net
pred = alex_net(x, weights, biases, keep_prob)

# define the loss function and learnning rate
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# test net
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# initialize
init = tf.initialize_all_variables()

# Add ops to save and restore all the variables
saver = tf.train.Saver()



# begin to train
with tf.Session() as sess:
    sess.run(init)

    merged_summary_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter('./alexnet_logs', sess.graph)

    step = 1
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        #batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        batch_xs = train_data_batch
        batch_ys = train_label_batch
        # get a batch
        sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys, keep_prob: dropout})
        if step % display_step == 0:
            # calculate the accuracy
            acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
            # calculate the loss
            loss = sess.run(cost, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
            print ("Iter " + str(step*batch_size) + ", Minibatch Loss= " + "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(acc))

        step += 1
    summary_str = sess.run(merged_summary_op)
    summary_writer.add_summary(summary_str, step)
    print ("Optimization Finished!")
    # calculate the test accuracy
    #print ("Testing Accuracy:", sess.run(accuracy, feed_dict={x: mnist.test.images[:256], y: mnist.test.labels[:256], keep_prob: 1.}))