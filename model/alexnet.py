import tensorflow as tf
import os
import get_input_data as input
from datetime import datetime
import numpy as np

#x = tf.placeholder(tf.float32, [batch_size, 227, 227, 3])
#y = tf.placeholder(tf.float32, [None, num_classes])

class network(object):

    def inference(self,images):

      #  images = tf.reshape(images, shape=[-1, 39,39, 3])

        images = tf.reshape(images, shape=[-1, 227,227, 3])# [batch, in_height, in_width, in_channels]

        #images=(tf.cast(images,tf.float32)/255.-0.5)*2


        # first layer

        conv1=tf.nn.bias_add(tf.nn.conv2d(images, self.weights['conv1'], strides=[1, 4, 4, 1], padding='VALID'),

                             self.biases['conv1'])



        relu1= tf.nn.relu(conv1)

        pool1=tf.nn.max_pool(relu1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')

        # second layer
        conv2=tf.nn.bias_add(tf.nn.conv2d(pool1, self.weights['conv2'], strides=[1, 1, 1, 1], padding='SAME'),

                             self.biases['conv2'])
        relu2= tf.nn.relu(conv2)
        pool2=tf.nn.max_pool(relu2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')

        # thrid layer
        conv3=tf.nn.bias_add(tf.nn.conv2d(pool2, self.weights['conv3'], strides=[1, 1, 1, 1], padding='SAME'),

                             self.biases['conv3'])
        relu3= tf.nn.relu(conv3)

        # fourth layer
        conv4=tf.nn.bias_add(tf.nn.conv2d(relu3, self.weights['conv4'], strides=[1, 1, 1, 1], padding='SAME'),

                             self.biases['conv4'])
        relu4= tf.nn.relu(conv4)

        # fifth layer
        conv5=tf.nn.bias_add(tf.nn.conv2d(relu4, self.weights['conv5'], strides=[1, 1, 1, 1], padding='SAME'),

                             self.biases['conv5'])

        relu5= tf.nn.relu(conv5)
        pool5=tf.nn.max_pool(relu5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')

        # fc6
        flatten = tf.reshape(pool5, [-1, self.weights['fc1'].get_shape().as_list()[0]])
        drop1=tf.nn.dropout(flatten,0.5)
        fc1=tf.matmul(drop1, self.weights['fc1'])+self.biases['fc1']
        fc_relu1=tf.nn.relu(fc1)

        # fc7
        fc2 = tf.matmul(fc_relu1, self.weights['fc2'])+self.biases['fc2']
        fc_relu2 = tf.nn.relu(fc2)

        # fc8
        fc3 = tf.matmul(fc_relu2, self.weights['fc3'])+self.biases['fc3']

        return fc3

    def __init__(self):

        # init weight and b

        with tf.variable_scope("weights"):

           self.weights={

                #39*39*3->36*36*20->18*18*20

                'conv1':tf.get_variable('conv1',[11,11,3,96],initializer=tf.contrib.layers.xavier_initializer_conv2d()),

                #18*18*20->16*16*40->8*8*40

                'conv2':tf.get_variable('conv2',[5,5,96,256],initializer=tf.contrib.layers.xavier_initializer_conv2d()),

                #8*8*40->6*6*60->3*3*60

                'conv3':tf.get_variable('conv3',[3,3,256,384],initializer=tf.contrib.layers.xavier_initializer_conv2d()),

                #3*3*60->120

                'conv4':tf.get_variable('conv4',[3,3,384,384],initializer=tf.contrib.layers.xavier_initializer_conv2d()),



                'conv5':tf.get_variable('conv5',[3,3,384,256],initializer=tf.contrib.layers.xavier_initializer_conv2d()),





                'fc1':tf.get_variable('fc1',[6*6*256,4096],initializer=tf.contrib.layers.xavier_initializer()),

                'fc2':tf.get_variable('fc2',[4096,4096],initializer=tf.contrib.layers.xavier_initializer()),



                #120->6

                'fc3':tf.get_variable('fc3',[4096,23],initializer=tf.contrib.layers.xavier_initializer()),

                }

        with tf.variable_scope("biases"):

            self.biases={

                'conv1':tf.get_variable('conv1',[96,],initializer=tf.constant_initializer(value=0.0, dtype=tf.float32)),

                'conv2':tf.get_variable('conv2',[256,],initializer=tf.constant_initializer(value=0.0, dtype=tf.float32)),

                'conv3':tf.get_variable('conv3',[384,],initializer=tf.constant_initializer(value=0.0, dtype=tf.float32)),

                'conv4':tf.get_variable('conv4',[384,],initializer=tf.constant_initializer(value=0.0, dtype=tf.float32)),

                'conv5':tf.get_variable('conv5',[256,],initializer=tf.constant_initializer(value=0.0, dtype=tf.float32)),



                'fc1':tf.get_variable('fc1',[4096,],initializer=tf.constant_initializer(value=0.0, dtype=tf.float32)),

                'fc2':tf.get_variable('fc2',[4096,],initializer=tf.constant_initializer(value=0.0, dtype=tf.float32)),

                'fc3':tf.get_variable('fc3',[23,],initializer=tf.constant_initializer(value=0.0, dtype=tf.float32))



            }


    def inference_test(self,images):

        # turn vector to matrix

        images = tf.reshape(images, shape=[-1, 39,39, 3])# [batch, in_height, in_width, in_channels]

        images=(tf.cast(images,tf.float32)/255.-0.5)*2


        # first layer

        conv1=tf.nn.bias_add(tf.nn.conv2d(images, self.weights['conv1'], strides=[1, 1, 1, 1], padding='VALID'),

                             self.biases['conv1'])



        relu1= tf.nn.relu(conv1)

        pool1=tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')


        # second layer

        conv2=tf.nn.bias_add(tf.nn.conv2d(pool1, self.weights['conv2'], strides=[1, 1, 1, 1], padding='VALID'),

                             self.biases['conv2'])

        relu2= tf.nn.relu(conv2)

        pool2=tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')


        # third layer

        conv3=tf.nn.bias_add(tf.nn.conv2d(pool2, self.weights['conv3'], strides=[1, 1, 1, 1], padding='VALID'),

                             self.biases['conv3'])

        relu3= tf.nn.relu(conv3)

        pool3=tf.nn.max_pool(relu3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')





        # fully connected layers 1

        flatten = tf.reshape(pool3, [-1, self.weights['fc1'].get_shape().as_list()[0]])



        fc1=tf.matmul(flatten, self.weights['fc1'])+self.biases['fc1']

        fc_relu1=tf.nn.relu(fc1)


        fc2=tf.matmul(fc_relu1, self.weights['fc2'])+self.biases['fc2']

        return fc2

    def sorfmax_loss(self, predicts, labels):

        predicts=tf.nn.softmax(predicts)

        labels=tf.one_hot(labels,self.weights['fc3'].get_shape().as_list()[1])

        loss = tf.nn.softmax_cross_entropy_with_logits(logits=predicts, labels=labels)

      #  loss =-tf.reduce_mean(labels * tf.log(predicts))# tf.nn.softmax_cross_entropy_with_logits(predicts, labels)

        self.cost= tf.reduce_mean(loss)

        return self.cost

    # gradient
    def optimer(self,loss,lr=0.01):

        train_optimizer = tf.train.GradientDescentOptimizer(lr).minimize(loss)

        return train_optimizer

    # calculate accuracy
    def accuracy(self, predicts, labels):

        acc = tf.nn.in_top_k(predicts, labels, 1)

        acc = tf.cast(acc, tf.float32)

        acc = tf.reduce_mean(acc)

        '''
        labels = tf.one_hot(labels, self.weights['fc3'].get_shape().as_list()[1])
        correct_pred = tf.equal(tf.argmax(predicts, 1), tf.argmax(labels, 1))
        acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        
        '''

        return acc




def train():

    root_dir = "/home/alala/Projects/part_constellation_models_tensorflow"

    # params
    batch_size = 64
    num_classes = 23
    num_epochs = 20

    # How often we want to write the tf.summary data to disk
    display_step = 20
    max_iter = 10000

    # Path for tf.summary.FileWriter and to store model checkpoints
    filewriter_path = root_dir + "/finetune_alexnet/plane23"
    checkpoint_path = root_dir + "/finetune_alexnet/"

    # Create parent path if it doesn't exist
    if not os.path.isdir(checkpoint_path): os.mkdir(checkpoint_path)
    if not os.path.isdir(filewriter_path): os.mkdir(filewriter_path)

    # create the network
    net = network()

    # get data
    train_dir = root_dir + "/data/plane23/train"
    image_list, label_list=input.get_data_list(train_dir)
    batch_image,batch_label = input.get_batch(image_list,label_list,227,227,64,256)

    inf = net.inference(batch_image)

    # calculate loss
    loss=net.sorfmax_loss(inf,batch_label)
    opti=net.optimer(loss)

    # test net
    accuracy = net.accuracy(inf, batch_label)

    # Merge all summaries together
    merged_summary = tf.summary.merge_all()

    # Initialize the FileWriter
    writer = tf.summary.FileWriter(filewriter_path)

    # Initialize an saver for store model checkpoints
    saver = tf.train.Saver()

    init = tf.initialize_all_variables()

    with tf.Session() as sess:

        sess.run(init)

        coord = tf.train.Coordinator()

        threads = tf.train.start_queue_runners(coord=coord)

        # Add the model graph to TensorBoard
        writer.add_graph(sess.graph)

        print("{} Start training...".format(datetime.now()))
        print("{} Open Tensorboard at --logdir {}".format(datetime.now(),filewriter_path))

        try:
            # Loop over number of epochs
            for epoch in range(num_epochs):

                #image_list, label_list = input.get_data_list(train_dir)
                #batch_image, batch_label = input.get_batch(image_list, label_list, 227, 227, 64, 256)

                train_batches_per_epoch = np.floor(len(image_list) / batch_size).astype(np.int16)

                print("{} Epoch number: {}".format(datetime.now(), epoch + 1))

                step = 1

                # train_batches_per_epoch
                while step < max_iter:

                    loss_np, opti_np = sess.run([loss, opti])

                    # Generate summary with the current batch of data and write to file
                    if step % display_step == 0:

                        acc = sess.run([accuracy])

                        print "Iter " + str(step * batch_size) + ", Minibatch Loss= " + "{:.6f}".format(loss_np) + ", Training Accuracy= " + "{:.5f}".format(acc[0])

                    #s = sess.run([merged_summary, batch_image, batch_label, inf])
                    #writer.add_summary(s, epoch * train_batches_per_epoch + step)

                    step += 1

                print("{} Saving checkpoint of model...".format(datetime.now()))

                # save checkpoint of the model
                checkpoint_name = os.path.join(checkpoint_path, 'model_epoch' + str(epoch) + '.ckpt')
                saver.save(sess, checkpoint_name)

                print("{} Model checkpoint saved at {}".format(datetime.now(), checkpoint_name))

        except tf.errors.OutOfRangeError:

            print ("Optimization Finished!")

        finally:

            coord.request_stop()

        coord.join(threads)

        sess.close()


if __name__ == '__main__':

    train()
