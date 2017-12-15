import numpy as np
import cv2
import os
import random


class GetPlaneImage:
    def __init__(self, dtype, mode=None, imagedir=None, labeldir=None, tr_ID_file=None, class_list=None,
                 flip=False, rotate=False, shuffle=False, mask=False, crop=False,
                 mean=np.array([124., 117., 104.]), scale_size=(224, 224),
                 nb_classes=200):

        # Init params
        self.imagedir = imagedir
        self.labeldir = labeldir
        self.tr_ID_file = tr_ID_file
        self.class_list = class_list

        self.flip = flip
        self.rotate = rotate
        self.shuffle = shuffle
        self.crop = crop
        self.mask = mask

        self.n_classes = nb_classes
        self.mean = mean
        self.scale_size = scale_size
        self.pointer = 0

        self.read_class_list(dtype, mode)

        if self.flip:
            self.do_flip()

        # self.get_mean()

        if self.shuffle:
            self.shuffle_data()

    def read_class_list(self, dtype, mode):
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

            if mode == 'train':
                index = np.where(tr_ID == 1)
            elif mode == 'test':
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

    def do_flip(self):
        fliped_img_path = '/home/alala/Projects/part_constellation_models_tensorflow/data/CUB_200_2011/flip/'
        if not os.path.isdir(fliped_img_path): os.mkdir(fliped_img_path)

        for i in range(len(self.images)):
            img = cv2.imread(self.images[i])

            # flip with y
            img_f = cv2.flip(img, 1)
            path_f = fliped_img_path + 'fliped_' + str(i) + '.jpg'
            self.images.append(path_f)
            self.labels.append(self.labels[i])
            cv2.imwrite(path_f, img_f)

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

    def next_batch(self, batch_size, showImg=False):
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
            #cv2.imshow('original image', img)

            # crop the image randomly
            if self.crop and np.random.random() < 0.5:
                imgsize = img.shape
                left_x = int(random.uniform(0, 20) / 100 * imgsize[1])
                left_y = int(random.uniform(0, 20) / 100 * imgsize[0])
                right_y = int(random.uniform(80, 100) / 100 * imgsize[0]) - 1
                right_x = int(random.uniform(80, 100) / 100 * imgsize[1]) - 1
                img = img[left_y:right_y, left_x:right_x]
                #cv2.imshow('random crop', img)

            # generate mask for image
            if self.mask and np.random.random() < 0.5:
                choice = np.random.random()
                if 0 <= choice < 0.5:
                    imgsize = img.shape
                    random_w = int(random.uniform(0, 30) / 100 * imgsize[1])
                    random_h = int(random.uniform(0, 30) / 100 * imgsize[0])
                    block_w = (random_w if random_w > 40 else 40)
                    block_h = (random_h if random_h > 40 else 40)

                    left_x = int(random.uniform(15, 85) / 100 * imgsize[1])
                    left_y = int(random.uniform(15, 85) / 100 * imgsize[0])
                    right_x = (int(imgsize[1])-1 if left_x+block_w >= int(imgsize[1]) else left_x+block_w)
                    right_y = (int(imgsize[0])-1 if left_y+block_h >= int(imgsize[0]) else left_y+block_h)

                    mask_flag = 0
                    # set the area to black
                    img[left_y:right_y, left_x:right_x, :] = 0
                    #cv2.imshow('random mask', img)
                else:
                    imgsize = img.shape
                    random_w = int(random.uniform(40, 70) / 100 * imgsize[1])
                    random_h = int(random.uniform(40, 70) / 100 * imgsize[0])
                    block_w = (random_w if random_w > 100 else 100)
                    block_h = (random_h if random_h > 100 else 100)

                    left_x = int(random.uniform(20, 50) / 100 * imgsize[1])
                    left_y = int(random.uniform(20, 50) / 100 * imgsize[0])
                    right_x = (int(imgsize[1]) - 1 if left_x + block_w >= int(imgsize[1]) else left_x + block_w)
                    right_y = (int(imgsize[0]) - 1 if left_y + block_h >= int(imgsize[0]) else left_y + block_h)

                    # set the area to black
                    show_area = np.zeros([imgsize[0], imgsize[1], 3])
                    show_area[left_y:right_y, left_x:right_x, :] = 1

                    img = (img * show_area).astype('uint8')

                    #cv2.imshow('mask all image', img)

            # rotate the image randomly
            if self.rotate and np.random.random() < 0.5:
                imgsize = img.shape
                rotate_angles = random.uniform(-20, 20)
                m = cv2.getRotationMatrix2D((imgsize[1] / 2, imgsize[0] / 2), rotate_angles, 1)
                img = cv2.warpAffine(img, m, (imgsize[1], imgsize[0]))
                #cv2.imshow('random rotate', img)

            #cv2.imshow('input_image', img)

            # rescale image
            img = cv2.resize(img, (self.scale_size[0], self.scale_size[1]))
            img = img.astype(np.float32)

            # subtract mean
            img -= self.mean

            images[i] = img

            img = []

        # Expand labels to one hot encoding
        #one_hot_labels = np.zeros((batch_size, self.n_classes))
        #for i in range(len(labels)):
            #one_hot_labels[i][labels[i]] = 1

            # return array of images and labels
        one_hot_labels = labels

        return images, one_hot_labels
