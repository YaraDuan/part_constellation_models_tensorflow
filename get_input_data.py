import numpy as np
import os

train_dir = './data/train'

def train_data(file_dir):

    for file in os.listdir(file_dir+'/Cat'):
            cats.append(file_dir +'/Cat'+'/'+ file)
            label_cats.append(0)
    for file in os.listdir(file_dir+'/Dog'):
            dogs.append(file_dir +'/Dog'+'/'+file)
            label_dogs.append(1)

    # ?cat?dog???????list?img?lab?
    image_list = np.hstack((cats, dogs))
    label_list = np.hstack((label_cats, label_dogs))

    # ??shuffle????
    temp = np.array([image_list, label_list])
    temp = temp.transpose()
    np.random.shuffle(temp)

    # ????temp????list?img?lab?
    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])
    label_list = [int(i) for i in label_list]