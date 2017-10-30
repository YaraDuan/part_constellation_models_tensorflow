import tensorflow as tf

import get_input_data as inputdata

from model import alexnet

crop_w = crop_h = 227
batch_size = 64
capacity = 256

# load data
train_dir = './data/plane23/train'
train_data, train_label = inputdata.get_data_list(train_dir)
train_data_batch, train_label_batch = inputdata.get_batch(train_data,train_label,crop_w,crop_h,batch_size,capacity)

alexnet.train(train_data_batch,train_label_batch)