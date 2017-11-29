import tensorflow as tf
from vgg19 import *
from get_input_data2 import GetPlaneImage
import numpy as np


# train
def train(batch_size, save_path, dropoutPro):

    root_dir = "/home/alala/Projects/part_constellation_models_tensorflow"

    # params
    num_epochs = 100000

    # How often we want to write the tf.summary data to disk
    display_step = 20
    max_iter = 10000

    # Path for tf.summary.FileWriter and to store model checkpoints
    filewriter_path = root_dir + "/model/vgg19/checkpoints/" + save_path
    checkpoint_path = root_dir + "/model/vgg19/checkpoints/" + save_path

    # Create parent path if it doesn't exist
    if not os.path.isdir(filewriter_path) : os.mkdir(filewriter_path)
    if not os.path.isdir(checkpoint_path): os.mkdir(checkpoint_path)

    x = tf.placeholder(tf.float32, [batch_size, crop_w, crop_h, 3])
    y = tf.placeholder(tf.int32, [batch_size, classNum])

    # create the network
    net = VGG19(x, dropoutPro, classNum, skip)

    # calculate loss
    loss = net.get_loss(net.vgg19_net['fc8'], y)
    opti = net.optimer(loss)

    # test net
    accuracy = net.accuracy(net.vgg19_net['fc8'], y)

    # Initialize an saver for store model checkpoints
    saver = tf.train.Saver()

    init = tf.global_variables_initializer()

    config = tf.ConfigProto(allow_soft_placement=True)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)

    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:

        sess.run(init)

        coord = tf.train.Coordinator()

        threads = tf.train.start_queue_runners(coord=coord)

        # Merge all summaries together
        merged_summary = tf.summary.merge_all()

        # Initialize the FileWriter
        writer = tf.summary.FileWriter(filewriter_path, sess.graph)

        print("{} Start training...".format(datetime.now()))
        print("{} Open Tensorboard at --logdir {}".format(datetime.now(), filewriter_path))

        try:
            # Loop over number of epochs
            for epoch in range(1, num_epochs+1):

                train_batches_per_epoch = np.floor(train_data.data_size / batch_size).astype(np.int16)

                print("{} Epoch number: {}".format(datetime.now(), epoch))

                step = 1

                # train_batches_per_epoch
                while step < train_batches_per_epoch+1:

                    images, labels = train_data.next_batch(batch_size)

                    sess.run(opti, feed_dict={x: images, y: labels})

                    iters = (train_batches_per_epoch * batch_size) * (epoch - 1) + step * batch_size

                    # Generate summary with the current batch of data and write to file
                    if step % display_step == 0:

                        loss_np = sess.run(loss, feed_dict={x: images, y: labels})

                        acc = sess.run(accuracy, feed_dict={x: images, y: labels})

                        print "Iter " + str(iters) + ", Minibatch Loss= " + "{:.6f}".format(
                            loss_np) + ", Training Accuracy= " + "{:.5f}".format(acc)

                    s = sess.run(merged_summary, feed_dict={x: images, y: labels})
                    writer.add_summary(s, iters)

                    step += 1

                # save checkpoint of the model
                if epoch % 20 == 0:
                    checkpoint_name = os.path.join(checkpoint_path, 'model_epoch' + str(epoch) + '.ckpt')
                    saver.save(sess, checkpoint_name)
                    print("{} Model checkpoint saved at {}".format(datetime.now(), checkpoint_name))

                    # test the model
                    test_acc = 0
                    count = 0
                    test_batches_per_epoch = np.floor(test_data.data_size / batch_size).astype(np.int16)
                    for i in range(test_batches_per_epoch):
                        test_images, test_labels = test_data.next_batch(batch_size)
                        acc = sess.run(accuracy, feed_dict={x: test_images, y: test_labels})
                        test_acc += acc
                        count += 1

                    test_acc = test_acc/count
                    print "{}".format(datetime.now()) + " Test Accuracy= " + "{:.5f}".format(test_acc)

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
    batch_size = 32

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
    train_data = GetPlaneImage(dtype='CUB_200_2011', model='train', imagedir=image_txt, labeldir=label_txt, tr_ID_file=tr_ID_txt, mean=mean, shuffle=True, nb_classes=classNum)

    test_data = GetPlaneImage(dtype='CUB_200_2011', model='test', imagedir=image_txt, labeldir=label_txt, tr_ID_file=tr_ID_txt, mean=mean, shuffle=True, nb_classes=classNum)

    # get plane images
    """
    train_dir = root_dir + "/data/plane23/train.txt"
    input_data = GetPlaneImage(dtype='plane', class_list=train_dir, shuffle=True, mean=np.array([103.438, 113.549, 114.933]), nb_classes=classNum)
    data_size = input_data.data_size
    """
    train(batch_size, save_path, dropoutPro)