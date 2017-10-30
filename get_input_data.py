import numpy as np
import os
import tensorflow as tf


# get the name and id of labels
label_file = open('/home/alala/Projects/part_constellation_models_tensorflow/data/plane23/label.txt','r')
lines = label_file.readlines()
labels = []
imgdir_list = []

for line in lines:
    label_id = line.split()[0]
    label_name = line.split()[1]
    labels.append([label_id, label_name])

def train_data(file_dir, crop_w, crop_h):

    image_list = []
    label_list = []

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
                if file.startswith('.') and os.path.isfile(os.path.join(imgpath,file)):
                    files.remove(file)

            files.sort(key=lambda x: int(x[:-4]))
            for file in files:
                image_list.append(os.path.join(imgpath, file))
                label_list.append(class_num)

    # use shuffle to disrupt the order
    temp = np.array([image_list, label_list])
    temp = temp.transpose()
    np.random.shuffle(temp)

    # get image_list and label list
    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])
    label_list = [int(i) for i in label_list]

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

    return image_list, label_list


if __name__ == '__main__':
    train_dir = './data/plane23/train'
    train_data(train_dir, 227, 227)