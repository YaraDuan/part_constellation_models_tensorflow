import tensorflow as tf
from vgg19 import *
from get_input_data2 import GetPlaneImage
import numpy as np


# train
def train(batch_size, save_path, dropoutPro, data_size):

    root_dir = "/home/alala/Projects/part_constellation_models_tensorflow"

    # params
    num_epochs = 100000

    # How often we want to write the tf.summary data to disk
    display_step = 20
    max_iter = 10000

    # Path for tf.summary.FileWriter and to store model checkpoints
    filewriter_path = root_dir + "/model/vgg19/checkpoints/" + save_path
    checkpoint_path = root_dir + "/model/vgg19/checkpoints/" + save_path +"/models"

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

                train_batches_per_epoch = np.floor(data_size / batch_size).astype(np.int16)

                print("{} Epoch number: {}".format(datetime.now(), epoch))

                step = 1

                # train_batches_per_epoch
                while step < train_batches_per_epoch:

                    images, labels = input_data.next_batch(batch_size)

                    #images, labels = sess.run([batch_image, batch_label])

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
    """
    # get data
    train_dir = root_dir + "/data/plane23/train"
    image_list, label_list = input.get_data_list('plane23', train_dir)
    batch_image, batch_label = input.get_batch(image_list, label_list, crop_w, crop_h, batch_size, 256)
    """

    image_txt = root_dir + "/data/CUB_200_2011/imagelist_absolute.txt"
    label_txt = root_dir + "/data/CUB_200_2011/labels.txt"
    """
    image_list, label_list = input.get_data_list('CUB_200_2011', image_txt, label_txt)
    batch_image, batch_label = input.get_batch(image_list, label_list, crop_w, crop_h, batch_size, 256)
    """

    # get image
    input_data = GetPlaneImage(dtype='CUB_200_2011', imagedir=image_txt, labeldir=label_txt, shuffle=True, nb_classes=classNum)
    data_size = input_data.data_size
    """
    train_dir = root_dir + "/data/plane23/train.txt"
    input_data = GetPlaneImage(dtype='plane', class_list=train_dir, shuffle=True, mean=np.array([103.438, 113.549, 114.933]), nb_classes=classNum)
    data_size = input_data.data_size
    """
    train(batch_size, save_path, dropoutPro, data_size)