import tensorflow as tf
from vgg19 import *
from get_input_data2 import GetPlaneImage
import numpy as np


# train
def train(batch_size, save_path, dropoutPro):

    root_dir = "/home/alala/Projects/part_constellation_models_tensorflow"

    # finetune on pretrained model
    VGG_MODEL = 'imagenet-vgg-verydeep-19.mat'

    # params
    num_epochs = 100000
    display_step = 100
    lr = 0.0001

    test_batch_size = 8

    # Path for tf.summary.FileWriter and to store model checkpoints
    filewriter_path = root_dir + "/model/vgg19/checkpoints/" + save_path
    checkpoint_path = root_dir + "/model/vgg19/checkpoints/" + save_path

    meta_path = checkpoint_path + '/model_epoch20.ckpt.meta'
    model_path = checkpoint_path + '/model_epoch20.ckpt'

    # Create parent path if it doesn't exist
    if not os.path.isdir(filewriter_path): os.mkdir(filewriter_path)
    if not os.path.isdir(checkpoint_path): os.mkdir(checkpoint_path)

    x = tf.placeholder(tf.float32, [batch_size, crop_w, crop_h, 3], name="input_x")
    y = tf.placeholder(tf.int32, [batch_size], name="input_y")

    # create the network
    net = VGG19(x, dropoutPro, classNum, skip)

    # calculate loss
    loss = net.get_loss(net.vgg19_net['softmax'], y)
    opti = net.optimer(loss, lr=lr)

    # test net
    accuracy_top_1 = net.accuracy_top_1(net.vgg19_net['fc8'], y)
    accuracy_top_5 = net.accuracy_top_5(net.vgg19_net['fc8'], y)

    # Initialize an saver for store model checkpoints
    saver = tf.train.Saver(max_to_keep=100)

    init = tf.global_variables_initializer()

    # set 70% memory of GPU to run
    config = tf.ConfigProto(allow_soft_placement=True)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
    config.gpu_options.allow_growth = True

    result_train = np.zeros([classNum, classNum])
    result_test = np.zeros([classNum, classNum])

    with tf.Session(config=config) as sess:

        sess.run(init)

        coord = tf.train.Coordinator()

        threads = tf.train.start_queue_runners(coord=coord)

        # saver.restore(sess, model_path)

        net.loadModel(sess, model_mode='MATLAB', modelpath=VGG_MODEL)

        # Merge all summaries together
        merged_summary = tf.summary.merge_all()

        # Initialize the FileWriter
        train_writer = tf.summary.FileWriter(filewriter_path + '/train', sess.graph)
        train_writer2 = tf.summary.FileWriter(filewriter_path + '/train2')
        test_writer = tf.summary.FileWriter(filewriter_path + '/test')

        print("{} Start training...".format(datetime.now()))
        print("{} Open Tensorboard at --logdir {}".format(datetime.now(), filewriter_path))
        w1=0
        b1=0

        try:
            # Loop over number of epochs
            for epoch in range(1, num_epochs+1):

                train_batches_per_epoch = np.floor(train_data.data_size / batch_size).astype(np.int16)

                print("{} Epoch number: {}".format(datetime.now(), epoch))

                step = 1

                avg_loss = 0
                avg_acc1 = 0
                avg_acc5 = 0

                # shuffle the data before train
                train_data.shuffle_data()

                # train_batches_per_epoch
                while step < train_batches_per_epoch+1:

                    images, labels = train_data.next_batch(batch_size)

                    summary, _, loss_np, acc1, acc5 = sess.run([merged_summary, opti, loss, accuracy_top_1, accuracy_top_5], feed_dict={x: images, y: labels})

                    '''
                    with tf.variable_scope('conv5_4', reuse=True):
                        w = tf.get_variable('w')
                        b = tf.get_variable('b')
                        grads_and_vars = tf.train.AdamOptimizer(lr).compute_gradients(loss, [w])
                        w0 = sess.run(w)
                        b0 = sess.run(b)

                    gr = sess.run([grad for grad, _ in grads_and_vars], feed_dict={x:images, y:labels})
                    print gr
                    print w1 == w0
                    print b1 == b0

                    w1 = w0
                    b1 = b0
                    '''
                    iters = (train_batches_per_epoch * batch_size) * (epoch - 1) + step * batch_size

                    #fc8, softmaxresult = sess.run([net.vgg19_net['fc8'], net.vgg19_net['softmax']],
                                                  #feed_dict={x: images, y: labels})

                    top1, top5 = sess.run(
                        [tf.nn.top_k(net.vgg19_net['softmax'], 1), tf.nn.top_k(net.vgg19_net['softmax'], 5)],
                        feed_dict={x: images, y: labels})

                    for m in range(batch_size):
                        for n in range(5):
                            result_train[labels[m]][top5.indices[m][n]] += top5.values[m][n]

                    avg_loss += loss_np

                    avg_acc1 += acc1
                    avg_acc5 += acc5

                    train_writer.add_summary(summary, iters)

                    # Generate summary with the current batch of data and write to file
                    if step % display_step == 0:

                        print "Iter " + str(iters) + ", Minibatch Loss= " + "{:.6f}".format(
                            loss_np) + ", Training Accuracy_Top1= " + "{:.5f}".format(
                            acc1) + ", Training Accuracy_Top5= " + "{:.5f}".format(
                            acc5) + "; Average Loss= " + "{:.6f}".format(
                            avg_loss/step) + ", Average Accuracy_Top1= " + "{:.5f}".format(
                            avg_acc1/step) + ", Average Accuracy_Top5= " + "{:.5f}".format(avg_acc5/step)

                    step += 1

                result_txt = open(root_dir + '/model/vgg19/result_train.txt', 'w')
                result_txt.truncate()
                for m in range(len(result_train)):
                    line = ''
                    for n in range(len(result_train)):
                        line += (str(result_train[m][n]) + ' ')
                    result_txt.writelines(line + '\n')
                result_txt.close()

                # calculate the loss and accuracy of the train data
                train2_loss = 0
                train2_acc1 = 0
                train2_acc5 = 0
                count2 = 0
                train2_batches_per_epoch = np.floor(480 / batch_size).astype(np.int16)
                train_data2.shuffle_data()
                for i in range(train2_batches_per_epoch):
                    train2_images, train2_labels = train_data2.next_batch(batch_size)
                    summary2, loss_np_train2, acc1, acc5 = sess.run([merged_summary, loss, accuracy_top_1, accuracy_top_5], feed_dict={x: train2_images, y: train2_labels})
                    train2_loss += loss_np_train2
                    train2_acc1 += acc1
                    train2_acc5 += acc5
                    count2 += 1

                train_writer2.add_summary(summary2, (train_batches_per_epoch * batch_size) * epoch)

                train2_acc1 = train2_acc1 / count2
                train2_acc5 = train2_acc5 / count2
                train2_loss = train2_loss / count2
                print "{}".format(datetime.now()) + " Train Loss= " + "{:.6f}".format(
                    train2_loss) + " Train Accuracy_TOP1= " + "{:.5f}".format(
                    train2_acc1) + " Train Accuracy_TOP2= " + "{:.5f}".format(train2_acc5)

                # test the model for each epoch
                test_loss = 0
                test_acc1 = 0
                test_acc5 = 0
                count = 0
                test_batches_per_epoch = np.floor(test_data.data_size / test_batch_size).astype(np.int16)
                for i in range(test_batches_per_epoch):
                    test_images, test_labels = test_data.next_batch(test_batch_size)
                    summary, loss_np_test, acc1, acc5 = sess.run([merged_summary, loss, accuracy_top_1, accuracy_top_5], feed_dict={x: test_images, y: test_labels})

                    top1, top5 = sess.run(
                        [tf.nn.top_k(net.vgg19_net['softmax'], 1), tf.nn.top_k(net.vgg19_net['softmax'], 5)],
                        feed_dict={x: test_images, y: test_labels})

                    for m in range(batch_size):
                        for n in range(5):
                            result_test[test_labels[m]][top5.indices[m][n]] += top5.values[m][n]

                    test_loss += loss_np_test
                    test_acc1 += acc1
                    test_acc5 += acc5
                    count += 1

                result_txt = open(root_dir + '/model/vgg19/result_test.txt', 'w')
                result_txt.truncate()
                for m in range(len(result_test)):
                    line = ''
                    for n in range(len(result_test)):
                        line += (str(result_test[m][n]) + ' ')
                    result_txt.writelines(line + '\n')
                result_txt.close()

                test_writer.add_summary(summary, (train_batches_per_epoch * batch_size) * epoch)

                test_acc1 = test_acc1 / count
                test_acc5 = test_acc5 / count
                test_loss = test_loss / count
                print "{}".format(datetime.now()) + " Test Loss= " + "{:.6f}".format(
                    test_loss) + " Test Accuracy_TOP1= " + "{:.5f}".format(test_acc1) + " Test Accuracy_TOP5= " + "{:.5f}".format(test_acc5)

                # save checkpoint of the model
                if epoch % 20 == 0:
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

    root_dir = "/home/alala/Projects/part_constellation_models_tensorflow"

    crop_h = 224
    crop_w = 224
    batch_size = 8

    # params
    dropoutPro = 0.5
    classNum = 200
    skip = []

    # the path to save model
    save_path = "CUB_200_2011"

    image_txt = root_dir + "/data/CUB_200_2011/imagelist_absolute.txt"
    label_txt = root_dir + "/data/CUB_200_2011/labels.txt"
    tr_ID_txt = root_dir + "/data/CUB_200_2011/tr_ID.txt"

    # get CUB_200_2011 images
    mean = np.array([124., 127., 110.])
    train_data = GetPlaneImage(dtype='CUB_200_2011', mode='train', imagedir=image_txt, labeldir=label_txt, tr_ID_file=tr_ID_txt,
                               mean=mean, shuffle=True, flip=True, crop=True, mask=True, rotate=True, nb_classes=classNum)

    train_data2 = GetPlaneImage(dtype='CUB_200_2011', mode='train', imagedir=image_txt, labeldir=label_txt, tr_ID_file=tr_ID_txt,
                                mean=mean, shuffle=True, nb_classes=classNum)

    test_data = GetPlaneImage(dtype='CUB_200_2011', mode='test', imagedir=image_txt, labeldir=label_txt, tr_ID_file=tr_ID_txt,
                              mean=mean, shuffle=True, nb_classes=classNum)

    # get plane images
    """
    train_dir = root_dir + "/data/plane23/train.txt"
    input_data = GetPlaneImage(dtype='plane', class_list=train_dir, shuffle=True, mean=np.array([103.438, 113.549, 114.933]), nb_classes=classNum)
    data_size = input_data.data_size
    """
    train(batch_size, save_path, dropoutPro)