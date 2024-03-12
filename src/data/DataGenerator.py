# -*- coding:utf-8 -*-
import math
import os

import cv2
import keras
import numpy as np
from keras.utils import to_categorical


class DataGenerator(keras.utils.Sequence):

    def __init__(self, x, param, y=None, batch_size=1, shuffle=True, preprocess=None, transform=None):
        self.batch_size = batch_size
        self.x = x
        self.y = y
        self.indexes = np.arange(len(self.x))
        self.shuffle = shuffle
        self.preprocess = preprocess
        self.transform = transform
        self.batch_size = batch_size
        self.param = param

        self.train_size = param.get_conf()['train_image_size']
        self.nb_class = self.param.get_conf()['classes_num']

    def __len__(self):
        # 计算每一个epoch的迭代次数
        return math.ceil(len(self.x) / float(self.batch_size))

    def __getitem__(self, index):
        # 生成每个batch数据，这里就根据自己对数据的读取方式进行发挥了
        # 生成batch_size个索引
        batch_indexs = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        # 根据索引获取datas集合中的数据
        batch_datas = [self.x[k] for k in batch_indexs]
        y = None
        if self.y is not None:
            y = [self.y[k] for k in batch_indexs]
        # 生成数据
        return self.data_generation(batch_datas, y)

    def on_epoch_end(self):
        # 在每一次epoch结束是否需要进行一次随机，重新随机一下index
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def data_generation(self, batch_datas, y=None):
        images = []
        labels = []
        if type(batch_datas[0]) is str:
            # print('self.data_path = ', self.data_path)
            # print('batch_datas = ', batch_datas)
            for i, data in enumerate(batch_datas):
                img = cv2.imread(data)[:, :, ::-1]
                img = cv2.resize(img, (self.train_size, self.train_size))
    
                images.append(img)

            # images = np.array(images)
        else:
            # images = np.array(batch_datas)
            images = batch_datas
        if self.transform is not None:
            for i, img in enumerate(images):
                images[i] = self.transform(img)
            # print(images.shape)
        images = np.array(images)
        if self.preprocess is not None:
            images = self.preprocess(images)
        if self.y is not None:
            for label in y:
                labels.append(to_categorical(label, self.nb_class))
            return np.array(images), np.array(labels)

        return np.array(images)
