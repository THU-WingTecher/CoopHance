# -*- coding:utf-8 -*-

from keras.datasets import cifar10
from subprocess import call
from data.data import Data
from utils import *
import scipy.io as scio

class SVHNData(Data):

    def __init__(self, param, is_adversarial=0):
        super(SVHNData, self).__init__(param)
        self.is_adversarial = is_adversarial
        if self.is_adversarial==0:
            self.data_path = os.path.join(self.param.get_conf('data_path'), 'train_32x32.mat')
            self.data_test_path = os.path.join(self.param.get_conf('data_path'), 'test_32x32.mat')
        elif self.is_adversarial == 1:
            self.data_path = os.path.join(self.param.get_conf('adversarial_dir'),
                                          self.param.get_conf('adversarial_type'))
            try:
                self.data_test_path = os.path.join(self.param.get_conf('adversarial_dir'),
                                          self.param.get_conf('cross_test'))
            except:
                self.data_test_path = self.data_path
        else:
            self.data_path = os.path.join(self.param.get_conf('adversarial_dir')[:-3],
                                          self.param.get_conf('adversarial_type'))
            self.data_test_path = self.data_path
        # print('is_adversarial', is_adversarial, self.data_path)

        self.n_class = self.param.get_conf('classes_num')
        self.x_train = []
        self.y_train = []
        self.x_test = []
        self.y_test = []
  
        self.random_selection_indices = None
        if self.is_adversarial != 0:
            self.load_adversarial()
        else:
            self.load_data()
            self.gen_shuffle_data()
        # self.gen_shuffle_train_data()

        self.n_class = self.param.get_conf('classes_num')

    def get_y_test_len(self):
        return len(self.y_test)

    def load_data(self):
        # print('self.data_path = ', self.data_path)

        if not self.is_adversarial:
            if not os.path.isfile(self.data_path):
                print('Downloading SVHN train set...')
                call(
                    "curl -o ../data/svhn/train_32x32.mat "
                    "http://ufldl.stanford.edu/housenumbers/train_32x32.mat",
                    shell=True
                )
            if not os.path.isfile(self.data_test_path):
                print('Downloading SVHN test set...')
                call(
                    "curl -o ../data/svhn/test_32x32.mat "
                    "http://ufldl.stanford.edu/housenumbers/test_32x32.mat",
                    shell=True
            )
            data_train = scio.loadmat(self.data_path)
            data_test = scio.loadmat(self.data_test_path)
            (self.x_train, self.y_train) = (data_train.get('X'), data_train.get('y'))
            (self.x_test, self.y_test) = (data_test.get('X'), data_test.get('y'))
            self.y_train = self.y_train.squeeze()
            self.y_test = self.y_test.squeeze()
            self.x_train = self.x_train.transpose((3,0,1,2))
            self.x_test = self.x_test.transpose((3,0,1,2))
            self.y_train -= 1
            self.y_test -= 1
            # serialize_img(self.x_test[0], self.param)

            # Add channel axis
            self.min_, self.max_ = 0, 255

            self.n_train = np.shape(self.x_train)[0]

        print('after reading data')
        print('x_train.shape = ', self.x_train.shape)
        print('y_train.shape = ', self.y_train.shape)
        print('x_test.shape = ', self.x_test.shape)
        print('y_test.shape = ', self.y_test.shape)




    def get_specific_label_data(self, label):
        y_train = self.y_train.argmax(axis=1)
        y_test = self.y_test.argmax(axis=1)
        return self.x_train[y_train == label], \
               self.y_train[y_train == label], \
               self.x_test[y_test == label], \
               self.y_test[y_test == label]

    def visiualize_img_by_idx(self, shuffled_idx, pre_label, is_train=True):
        pass

if __name__ == '__main__':
    json_name = sys.argv[1]
    param = Param(json_name)
    param.load_json()
    data = Data(param)
    data.load_data()
    data.gen_backdoor()
