# -*- coding:utf-8 -*-

import abc

from utils import *
from tqdm import tqdm
import copy
import math
from abc import ABCMeta, abstractmethod

class Adversarial(metaclass=abc.ABCMeta):
    def __init__(self, model, param):
        self.model = model
        self.param = param
        self.adversarial_type = self.param.get_conf('adversarial_type')
        self.save_root_path = os.path.join(self.param.get_conf('adversarial_dir'), self.adversarial_type)
        if not os.path.exists(self.save_root_path):
            os.makedirs(self.save_root_path)
        self.save_path = None
        self.iteration = None
        self.batch_size = math.ceil(param.get_conf("batch_size") / param.get_conf('num_gpu'))
        self.adv_perturbations = []
        self.adv_labels = []
        self.adv_indexes = []
        self.is_train = False

    def set_phase(self, is_train):
        self.is_train = is_train
        if is_train:
            self.save_path = os.path.join(self.save_root_path, 'train')
        else:
            self.save_path = os.path.join(self.save_root_path, 'test')
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def get_selection_rate(self):
        if 'train' in self.save_path:
            return self.param.get_conf("attack_rate")
        elif "test" in self.save_path:
            return self.param.get_conf("attack_rate_test")
        return 0

    def dump_img(self):
        if self.save_path is None:
            raise ("Please set the phase")
        path = self.save_path
        check_dir(path)
        if self.is_train:
            np.savez(path + '/{}_train_adv'.format(self.adversarial_type), imgs=self.adv_perturbations,
                     labels=self.adv_labels, idx=self.adv_indexes)
        else:
            np.savez(path + '/{}_test_adv'.format(self.adversarial_type), imgs=self.adv_perturbations,
                     labels=self.adv_labels, idx=self.adv_indexes)
        self.save_initialization()

    def save_img(self, img, label, idx):
        if len(img.shape) == 4:
            img = np.squeeze(img, axis=0)
        self.adv_perturbations.append(img)
        self.adv_labels.append(label)
        self.adv_indexes.append(idx)

    def save_initialization(self):
        self.adv_indexes = []
        self.adv_labels = []
        self.adv_perturbations = []

    # def save_img(self, img, label, idx):
    #     if self.save_path is None:
    #         raise ("Please set the phase")
    #     path = os.path.join(self.save_path, str(label))
    #     if not os.path.exists(path):
    #         os.makedirs(path)
    #     path = os.path.join(path, self.adversarial_type + "_" + str(idx))
    #     if len(img.shape) == 4:
    #         img = np.squeeze(img, axis=0)
    #     np.save(path + '.npy', img)
    #     img = image.array_to_img(img, scale=False)
    #     img.save(path + '.jpg')

    def predict(self, img):
        pred = self.model.predict_instance(img)
        label = np.argmax(pred[0])

        print('label = ', label)
        print('pred = ', pred)

        return label, pred

    def get_data(self, x, y=None,  idxs=None):
        if idxs is not None:
            if type(x) is list:
                x = [x[i] for i in idxs]
                
            else:
                x = x[idxs]
            if y is not None:
                y = y[idxs]  
        if type(x[0]) is str:
            imgs = []
            for f in x:
                victim_img = cv2.imread(f)[:, :, ::-1]
                victim_img = cv2.resize(
                    victim_img, (self.param.get_conf('train_image_size'),
                                    self.param.get_conf('train_image_size')))
                imgs.append(victim_img)
            imgs = np.array(imgs)
        else:
            imgs = x
        return imgs, y


    def attack_on_data(self, x, y, process=(None, None)):
        assert self.iteration is not None
        if process is not None and len(process) != 2:
            raise ('process need preprocess and deprocess')
        self.attack_idx = np.arange(len(x))
        # np.random.shuffle(self.attack_idx)
        num_selection = math.ceil(len(x) * self.get_selection_rate())
        tot_loop = 0
        num_tot = 0
        tot_pert_norm = 0
        len_tot = len(x)
        tot_succ = 0
        self.save_initialization()
        with tqdm(total=num_selection) as pbar:
            for i in range(0, len_tot, self.batch_size):
                up_bound = min(i +  self.batch_size, len_tot)
                vimtim_img, label = self.get_data(x, y, self.attack_idx[i:up_bound])
                if process[0] is not None:
                    victim_img = process[0](vimtim_img)
                pert_imgs, new_labels, loops, norms = self.attack_batch(victim_img,
                                                        label)

                for idx in range(len(vimtim_img)):
                    if loops[idx] == 0:
                        continue
                    if self.save_all or  loops[idx] > 0 and new_labels[idx] != label[idx]:
                        save_img = pert_imgs[idx]
                        if process[1] is not None:
                            save_img = process[1](save_img)
                        self.save_img(save_img, label[idx], self.attack_idx[i+idx])
                    pbar.update(1)
                    num_tot += 1
                    tot_loop += loops[idx]
                    tot_pert_norm += norms[idx]
                    if loops[idx] > 0 and new_labels[idx] != label[idx]:
                        tot_succ += 1
                    if num_tot == num_selection:
                        break
                
                if num_tot == num_selection:
                    break
        self.dump_img()
        print('average loop:', tot_loop * 1.0 / num_tot)
        print('average perturbation norm: ', tot_pert_norm / num_tot)
        print("rate of successful attack: %.2f%%, total attack images: %d" %
              (tot_succ * 100.0 / num_tot, num_tot))
            #   (tot_succ * 100.0 / num_selection, num_selection))
    
    
    @abstractmethod
    def attack_batch(self, x, label):
        pass
