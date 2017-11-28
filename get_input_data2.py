import numpy as np
import cv2


class GetPlaneImage:
    def __init__(self, dtype, imagedir=None, labeldir=None, class_list=None, horizontal_flip=False, shuffle=False,
                 mean=np.array([124., 117., 104.]), scale_size=(224, 224),
                 nb_classes=200):

        # Init params
        self.dtype = dtype
        self.imagedir = imagedir
        self.labeldir = labeldir
        self.horizontal_flip = horizontal_flip
        self.n_classes = nb_classes
        self.shuffle = shuffle
        self.mean = mean
        self.scale_size = scale_size
        self.pointer = 0

        self.read_class_list(class_list)

        if self.shuffle:
            self.shuffle_data()

    def read_class_list(self, class_list):
        """
        Scan the image file and get the image paths and labels
        """
        self.images = []
        self.labels = []

        if self.dtype == 'plane':

            with open(class_list) as f:
                lines = f.readlines()
                for l in lines:
                    items = l.split()
                    self.images.append(items[0])
                    self.labels.append(int(items[1]))

        elif self.dtype == 'CUB_200_2011':

            # get the path of images
            image_dir = open(self.imagedir, 'r')
            lines = image_dir.readlines()
            for line in lines:
                self.images.append(line.strip('\n'))
            image_dir.close()

            # get the labels of images
            labels_dir = open(self.labeldir, 'r')
            lines = labels_dir.readlines()
            for line in lines:
                # label begins at 0
                self.labels.append(int(line) - 1)
            labels_dir.close()

        # store total number of data
        self.data_size = len(self.labels)

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
        if self.pointer == len(self.labels):
            self.reset_pointer()
            paths = self.images[self.pointer:self.pointer + batch_size]
            labels = self.labels[self.pointer:self.pointer + batch_size]

            # update pointer
            self.pointer += batch_size

        elif self.pointer > len(self.labels):
            last_pointer = self.pointer - batch_size

            diff = self.pointer - len(self.labels)

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
