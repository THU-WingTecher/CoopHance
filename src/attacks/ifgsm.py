# -*- coding:utf-8 -*-
import copy
import math
from tensorflow.python.ops.linalg_ops import norm

from tqdm import tqdm

from attacks.adversarial_attack import Adversarial
from utils import *


class PGD(Adversarial):
    def __init__(self, model, param,data_max=1,data_min=0, save_all=True, max_steps=50, epsilon=0.3,):
        super(PGD, self).__init__(model, param)
        self.input_tensor = self.model.get_input_tensor() 
        self.before_softmax_tensor = self.model.get_output_bef_softmax()
        self.output_tensor = self.model.get_output_tensor()
        self.attack_idx = None
        self.max = data_max
        self.min = data_min
        # self.num_classes = self.output_tensor.shape[-1]
        self.num_classes = param.get_conf('classes_num')
        self.pred = K.function([self.input_tensor],
                               [self.before_softmax_tensor])
        self.inds = K.placeholder((None, self.num_classes), dtype='float32')
        self.loss = K.categorical_crossentropy(self.inds, self.before_softmax_tensor,from_logits= True)
        self.grads = K.gradients(self.loss, self.input_tensor)[0]
        self.pert = K.sign(self.grads)
        self.sess = K.get_session()
        # sign = K.sign(self.grads)
        self.save_all = save_all
        self.epsilon = epsilon
        self.pert_tensor = None
        self.max_steps=max_steps
        self.epsilon = K.constant(epsilon)
        self.pert_tensor = K.sign(self.grads) * self.epsilon
        self.pert_img = K.clip(self.input_tensor + self.pert_tensor, 0,1)
        self.iteration = K.function([self.input_tensor, self.inds], [self.pert_img])
    def attack(self,
               x_train,
               y_train,
               x_test,
               y_test,
               process=(None, None),
               epsilon=None,
               max_steps=None):
        print('implement pgd attack')
        K.set_learning_phase(0)
        if epsilon is not None:
            self.epsilon = K.constant(epsilon)
            self.pert_tensor = K.sign(self.grads) * self.epsilon
            self.pert_img = K.clip(self.input_tensor + self.pert_tensor, 0,1)
            self.iteration = K.function([self.input_tensor, self.inds], [self.pert_img])
        if max_steps is not None:
            self.max_steps = max_steps
        self.set_phase(True)
        self.attack_on_data(x_train, y_train, process=process)
        self.set_phase(False)
        self.attack_on_data(x_test, y_test, process=process)

    def attack_batch(self, x, label):
        min_iters = np.ones_like(label) * -1
        min_norms = np.zeros_like(label, dtype=np.float32)
        cur_idx_all = np.arange(len(x))
        all_adv_x = np.zeros_like(x, dtype=np.float32)
        pre_def_shape = np.prod(x.shape[1:])
        def update_all(x_test, y_test, tar, cur_iter):
            pred = self.pred([x_test])[0].argmax(axis=1)
            inds_success = np.where(pred != y_test)[0]
            inds_unsuccess = np.where(pred == y_test)[0]
            succ_idx_all = cur_idx_all[inds_success]
            min_iters[succ_idx_all] = cur_iter
            min_norms[succ_idx_all] = np.linalg.norm((x_test[inds_success] - \
                                     x[succ_idx_all]).reshape((len(inds_success), pre_def_shape)), axis=1)
            all_adv_x[succ_idx_all] = x_test[inds_success]
            return cur_idx_all[inds_unsuccess], x_test[inds_unsuccess], y_test[inds_unsuccess], tar[inds_unsuccess]
        adv_x = copy.deepcopy(x)
        loop_i = 0
        target = to_categorical(label, self.num_classes)
        y = copy.deepcopy(label)
        cur_idx_all, adv_x, y, target = update_all(adv_x, y, target, loop_i)

        while len(cur_idx_all)>0 and loop_i < self.max_steps:
            adv_x = self.iteration([adv_x, target])[0]
            loop_i += 1
            cur_idx_all, adv_x, y, target = update_all(adv_x, y, target , loop_i)
            # print(loop_i)
        if len(cur_idx_all) >0:
            min_iters[cur_idx_all] = loop_i
            min_norms[cur_idx_all] = np.linalg.norm((adv_x - \
                                     x[cur_idx_all]).reshape((len(cur_idx_all), pre_def_shape)), axis=1)
            all_adv_x[cur_idx_all] = adv_x
        return all_adv_x, self.pred([all_adv_x])[0].argmax(axis=1), \
                     min_iters, min_norms


    def attack_force(self, x, label):
        adv_x = x.copy()
        loop_i=0
        # target = to_categorical(label, 10)
        while loop_i < self.max_steps:
            adv_x = self.iteration([adv_x, label])[0]
            loop_i += 1
            # print(loop_i)
       
        return adv_x

    def predict(self, img):
        pred = self.model.predict_instance(img)
        label = np.argmax(pred[0])

        print('label = ', label)
        print('pred = ', pred)

        return label, pred


