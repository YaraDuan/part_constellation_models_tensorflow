import numpy as np
import os
import tensorflow as tf


# get the name and id of labels
label_file = open('/home/alala/Projects/part_constellation_models_tensorflow/data/plane23/label.txt', 'r')
#label_file = open('/Users/Alala/Projects/part_constellation_models_tensorflow/data/plane23/label.txt', 'r')
lines = label_file.readlines()
labels = []
imgdir_list = []

for line in lines:
    label_id = line.split()[0]
    label_name = line.split()[1]
    labels.append([label_id, label_name])


def get_data_list(dtype, file_dir, label_dir=None):

    if dtype == 'plane23':

        image_list = []
        label_list = []
        imgdir_list = []

        for dirpath, dirs, files in os.walk(file_dir):
            for dir in dirs:
                for label in labels:
                    if dir == label[1]:
                        imgdir_list.append([os.path.join(file_dir, dir), int(label[0]), label[1]])
                        break

        imgdir_list.sort(key=lambda x: x[1])

        for imginfo in imgdir_list:
            imgpath = imginfo[0]
            class_num = imginfo[1]

            for root, dirs, files in os.walk(imgpath):

                # delete .DS_Store
                for file in files:
                    if file.startswith('.') and os.path.isfile(os.path.join(imgpath, file)):
                        files.remove(file)

                files.sort(key=lambda x: int(x[:-4]))
                for file in files:
                    image_list.append(os.path.join(imgpath, file))
                    label_list.append(class_num)

    elif dtype == 'CUB_200_2011':
        image_list = []
        label_list = []

        # get the path of images
        image_dir = open(file_dir, 'r')
        lines = image_dir.readlines()
        for line in lines:
            image_list.append(line.strip('\n'))
        image_dir.close()

        # get the labels of images
        labels_dir = open(label_dir, 'r')
        lines = labels_dir.readlines()
        for line in lines:
            # label begins at 0
            label_list.append(int(line)-1)
        labels_dir.close()

    # use shuffle to disrupt the order
    temp = np.array([image_list, label_list])
    temp = temp.transpose()
    np.random.shuffle(temp)

    # get image_list and label list
    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])
    label_list = [int(i) for i in label_list]

    return image_list, label_list


def get_batch(image_list, label_list, crop_w, crop_h, batch_size, capacity):

    # change it's type
    image_list = tf.cast(image_list, tf.string)
    label_list = tf.cast(label_list, tf.int32)

    # make an input queue
    input_queue = tf.train.slice_input_producer([image_list, label_list])

    label_list = input_queue[1]
    # read image from a queue
    image_contents = tf.read_file(input_queue[0])

    # decode the image
    image_list = tf.image.decode_jpeg(image_contents, channels=3)

    # crop the image
    image_list = tf.image.resize_image_with_crop_or_pad(image_list, crop_w, crop_h)

    # sub mean
    image_list = tf.image.per_image_standardization(image_list)

    # get batch
    image_batch, label_batch = tf.train.batch([image_list, label_list],
                                              batch_size=batch_size,
                                              num_threads=32,
                                              capacity=capacity)

    label_batch = tf.reshape(label_batch, [batch_size])
    image_batch = tf.cast(image_batch, tf.float32)

    return image_batch, label_batch
