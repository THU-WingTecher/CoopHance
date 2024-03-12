# -*- coding:utf-8 -*-

import abc
import os
from conf import rand_seed
import numpy as np


class Data(metaclass=abc.ABCMeta):
    def __init__(self, param):
        self.param = param
        self.batch_size = None
        self.data_path = None
        self.data_test_path = None
        self.x_train = []
        self.y_train = []
        self.x_test = []
        self.y_test = []
        self.n_train = 0
        self.adversarial_idxs_train = []
        self.adversarial_idxs_test = []
    @abc.abstractmethod
    def load_data(self, is_add_channel=False):
        pass

    def gen_shuffle_data(self, seeds=None):
        if seeds is not None:
            np.random.seed(seeds)
        else:
            np.random.seed(rand_seed)
        shuffle_indics = np.arange(len(self.x_train))
        np.random.shuffle(shuffle_indics)
        self.x_train = self.x_train[shuffle_indics]
        self.y_train = self.y_train[shuffle_indics]
        shuffle_indics = np.arange(len(self.x_test))
        np.random.shuffle(shuffle_indics)
        self.x_test = self.x_test[shuffle_indics]
        self.y_test = self.y_test[shuffle_indics]

    @abc.abstractmethod
    def visiualize_img_by_idx(self, shuffled_idx, pre_label, is_train=True):
        pass

    def load_adversarial(self):
        assert (self.data_path is not None)
        train_path = os.path.join(self.data_path, 'train')
        test_path = os.path.join(self.data_test_path, 'test')
        # print(train_path, test_path)

        # if os.path.exists(train_path):
        #     dir_list = os.listdir(train_path)
        #     dir_list.sort()
        #     for cls in dir_list:
        #         dirs = os.listdir(os.path.join(train_path, cls))
        #         dirs.sort()
        #         for fi in dirs:
        #             if '.npy' in fi:
        #                 self.adversarial_idxs_train.append(int(fi.split('_')[-1][:-4]))
        #                 img = np.load(os.path.join(train_path, cls, fi))
        #                 self.x_train.append(img)
        #                 self.y_train.append(int(cls))
        #     print('num of train adversarial image:',len(self.x_train))
        # if os.path.exists(test_path):
        #     dir_list = os.listdir(test_path)
        #     dir_list.sort()
        #     for cls in dir_list:
        #         dirs = os.listdir(os.path.join(test_path, cls))
        #         dirs.sort()
        #         for fi in dirs:
        #             if '.npy' in fi:
        #                 self.adversarial_idxs_test.append(int(fi.split('_')[-1][:-4]))
        #                 img = np.load(os.path.join(test_path, cls, fi))
        #                 self.x_test.append(img)
        #                 self.y_test.append(int(cls))
        #     print('num of test adversarial image:', len(self.x_test))

        if os.path.exists(train_path):
            file_list = os.listdir(train_path)
            file_list.sort()
            for fi in file_list:
                if '.npz' in fi:
                    train_adv_data = np.load(os.path.join(train_path, fi))
                    self.adversarial_idxs_train = train_adv_data['idx']
                    self.x_train = train_adv_data['imgs']
                    self.y_train = train_adv_data['labels']
                    # print(self.adversarial_idxs_train)
            print('num of train adversarial image:', len(self.x_train))
        if os.path.exists(test_path):
            file_list = os.listdir(test_path)
            file_list.sort()
            for fi in file_list:
                if '.npz' in fi:
                    test_adv_data = np.load(os.path.join(test_path, fi))
                    self.adversarial_idxs_test = test_adv_data['idx']
                    self.x_test = test_adv_data['imgs']
                    self.y_test = test_adv_data['labels']
            print('num of test adversarial image:', len(self.x_test))

        self.x_train = np.array(self.x_train)
        self.y_train = np.array(self.y_train)
        self.x_test = np.array(self.x_test)
        self.y_test = np.array(self.y_test)
        self.n_train = len(self.x_train)
        print('aftering loading adversaril data')
        print('x_train.shape', self.x_train.shape)
        print('x_test.shape', self.x_test.shape)

    def filter_misclassified(self, model):
        train_preds, test_preds = model.predict(self)
        train_preds = (train_preds == self.y_train)
        test_preds = (test_preds == self.y_test)
        self.x_train = self.x_train[train_preds, ...]
        self.y_train = self.y_train[train_preds, ...]
        self.x_test = self.x_test[test_preds, ...]
        self.y_test = self.y_test[test_preds, ...]
    
    
