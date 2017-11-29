import numpy as np
import cv2
import os

class GetPlaneImage:
    def __init__(self, dtype, model=None, imagedir=None, labeldir=None, tr_ID_file=None, class_list=None,
                 horizontal_flip=False, shuffle=False,
                 mean=np.array([124., 117., 104.]), scale_size=(224, 224),
                 nb_classes=200):

        # Init params
        self.imagedir = imagedir
        self.labeldir = labeldir
        self.tr_ID_file = tr_ID_file
        self.class_list = class_list
        self.horizontal_flip = horizontal_flip
        self.n_classes = nb_classes
        self.shuffle = shuffle
        self.mean = mean
        self.scale_size = scale_size
        self.pointer = 0

        self.read_class_list(dtype, model)

        #self.get_mean()

        if self.shuffle:
            self.shuffle_data()

    def read_class_list(self, dtype, model):
        """
        Scan the image file and get the image paths and labels
        """
        all_images = []
        all_labels = []
        tr_ID = []

        self.images = []
        self.labels = []

        if dtype == 'plane':

            with open(self.class_list) as f:
                lines = f.readlines()
                for l in lines:
                    items = l.split()
                    all_images.append(items[0])
                    all_labels.append(int(items[1]))

        elif dtype == 'CUB_200_2011':

            # get the path of images
            image_dir = open(self.imagedir, 'r')
            lines = image_dir.readlines()
            for line in lines:
                all_images.append(line.strip('\n'))
            image_dir.close()

            # get the labels of images
            labels_dir = open(self.labeldir, 'r')
            lines = labels_dir.readlines()
            for line in lines:
                # label begins at 0
                all_labels.append(int(line) - 1)
            labels_dir.close()

            # get tr_ID
            for line in open(self.tr_ID_file, 'r').readlines():
                tr_ID.append(int(line))
            tr_ID = np.array(tr_ID)

            if model == 'train':
                index = np.where(tr_ID == 1)
            elif model == 'test':
                index = np.where(tr_ID == 0)

            for i in index[0]:
                self.images.append(all_images[i])
                self.labels.append(all_labels[i])

            # store total number of data
            self.data_size = len(self.labels)

    def get_mean(self):
        img_size = 224
        sum_r = 0
        sum_g = 0
        sum_b = 0
        count = 0

        for img_path in self.images:
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (img_size, img_size))
            sum_r = sum_r + img[:, :, 0].mean()
            sum_g = sum_g + img[:, :, 1].mean()
            sum_b = sum_b + img[:, :, 2].mean()
            count = count + 1

        sum_r = sum_r / count
        sum_g = sum_g / count
        sum_b = sum_b / count
        img_mean = [sum_r, sum_g, sum_b]
        self.mean = np.array(img_mean)

    def shuffle_data(self):
        """
        Random shuffle the images and labels
        """
        images = self.images
        labels = self.labels
        self.images = []
        self.labels = []

        # create list of permutated index and shuffle data accoding to list
        idx = np.random.permutation(len(labels))
        for i in idx:
            self.images.append(images[i])
            self.labels.append(labels[i])

    def reset_pointer(self):
        """
        reset pointer to begin of the list
        """
        self.pointer = 0

        if self.shuffle:
            self.shuffle_data()

    def next_batch(self, batch_size):
        """
        This function gets the next n ( = batch_size) images from the path list
        and labels and loads the images into them into memory
        """

        # prevent index out of bounds and read the data recursively
        if (self.pointer+batch_size) == len(self.labels):
            self.reset_pointer()
            paths = self.images[self.pointer:self.pointer + batch_size]
            labels = self.labels[self.pointer:self.pointer + batch_size]

            # update pointer
            self.pointer += batch_size

        elif (self.pointer+batch_size) > len(self.labels):
            last_pointer = self.pointer

            diff = self.pointer + batch_size - len(self.labels)

            paths = self.images[last_pointer:len(self.labels)]
            labels = self.labels[last_pointer:len(self.labels)]

            self.reset_pointer()

            for i in range(diff):
                paths.append(self.images[self.pointer])
                labels.append(self.labels[self.pointer])
                self.pointer += 1

        else:
            # Get next batch of image (path) and labels
            paths = self.images[self.pointer:self.pointer + batch_size]
            labels = self.labels[self.pointer:self.pointer + batch_size]
            # update pointer
            self.pointer += batch_size

        # Read images
        images = np.ndarray([batch_size, self.scale_size[0], self.scale_size[1], 3])
        for i in range(len(paths)):
            img = cv2.imread(paths[i])

            # flip image at random if flag is selected
            if self.horizontal_flip and np.random.random() < 0.5:
                img = cv2.flip(img, 1)

            # rescale image
            img = cv2.resize(img, (self.scale_size[0], self.scale_size[1]))
            img = img.astype(np.float32)

            # subtract mean
            img -= self.mean

            images[i] = img

        # Expand labels to one hot encoding
        one_hot_labels = np.zeros((batch_size, self.n_classes))
        for i in range(len(labels)):
            one_hot_labels[i][labels[i]] = 1

            # return array of images and labels

        return images, one_hot_labels

# test
"""
def getData():

    class_list = '/home/alala/Projects/part_constellation_models_tensorflow/data/plane23/train.txt'

    data = GetPlaneImage(class_list)

    data.pointer = 4938

    img, label = data.next_batch(64)

    print data.images


if __name__ == '__main__':

    getData()
"""
