# -*- coding:utf-8 -*-
import copy
import math

from tqdm import tqdm

from attacks.adversarial_attack import Adversarial
from utils import *


class Deepfool(Adversarial):
    def __init__(self, model, param, data_max=1, data_min=0):
        super(Deepfool, self).__init__(model, param)
        self.input_tensor = self.model.get_input_tensor()
        self.before_softmax_tensor = self.model.get_output_bef_softmax()
        self.output_tensor = self.model.get_output_tensor()
        self.attack_idx = None
        self.max = data_max
        self.min = data_min
        self.pred = K.function([self.input_tensor],
                               [self.before_softmax_tensor])
        self.grads = [[] for i in range(self.param.get_conf('classes_num'))]
        # dydxs = [K.gradients(self.before_softmax_tensor[..., i], self.input_tensor)[0] for i in
        #          trange(self.param.get_conf('classes_num'))]
        # self.grads = [K.function([self.input_tensor], [dydx_label]) for dydx_label in dydxs]

    def attack(self, x_train, y_train, x_test, y_test, overshoot = 0.02, 
                        process=(None, None), max_iter=150):
        print('implement deepfool attack')
        self.overshoot = overshoot
        self.max_iter = max_iter
        K.set_learning_phase(0)
        self.set_phase(True)
        self.attack_on_data(x_train, y_train, process=process)
        self.set_phase(False)
        self.attack_on_data(x_test, y_test, process=process)

    def attack_on_data(self, x, y, process=(None, None)):
        if process is not None and len(process) != 2:
            raise ('process need preprocess and deprocess')
        self.attack_idx = np.arange(len(x))
        
        # np.random.shuffle(self.attack_idx)
        num_selection = math.ceil(len(x) * self.get_selection_rate())
        tot_loop = 0
        num_suc = 0
        num_tot = 0
        tot_pert_norm = 0
        self.save_initialization()
        with tqdm(total=num_selection) as pbar:
            for idx in self.attack_idx:
                if type(x[idx]) is str:
                    victim_img = cv2.imread(x[idx])[:, :, ::-1]
                    victim_img = cv2.resize(
                        victim_img, (self.param.get_conf('train_image_size'),
                                     self.param.get_conf('train_image_size')))
                else:
                    victim_img = x[idx:idx + 1]
                if process[0] is not None:
                    victim_img = process[0](victim_img)
                _, loop_i, pert_img, new_label = self.attack_batch(
                    victim_img, y[idx])
                
                if loop_i == 0:
                    continue
                
                cur_pert_norm = np.linalg.norm((pert_img - victim_img)/(self.max-self.min))
                tot_pert_norm += cur_pert_norm
                if new_label != y[idx]:
                    num_suc += 1
                if process[1] is not None:
                    pert_img = process[1](pert_img)
                self.save_img(pert_img, y[idx], idx)
                
                tot_loop += loop_i
                num_tot += 1
                pbar.update(1)

                if num_tot == num_selection:
                    break
        self.dump_img()
        print('average loop:', tot_loop * 1 / num_tot)
        print("average perturbation norm: " ,tot_pert_norm / num_tot)
        print("rate of successful attack: %.2f%%, total attack is %d" %
              (num_suc * 100.0 / num_tot, num_tot))


    def attack_batch(self,
                 image,
                 label,
                 num_classes=20):
        """
           :param image: Image of size HxWx3
           :param f: feedforward function (input: images, output: values of activation BEFORE softmax).
           :param grads: gradient functions with respect to input (as many gradients as classes).
           :param num_classes: num_classes (limits the number of classes to test against, by default = 10)
           :param overshoot: used as a termination criterion to prevent vanishing updates (default = 0.02).
           :return: minimal perturbation that fools the classifier, number of iterations that it required, new estimated_label and perturbed image
        """
        label = int(label)
        input_shape = image.shape
        pert_image = copy.deepcopy(image)
        f_i = self.get_predict(image)
        scores = np.squeeze(f_i)
        order = np.argsort(scores)[::-1]
        num_classes = min(num_classes, len(order))
        order = order[:num_classes]
        predicted_label = order[0]

        # f_i = np.array(f).flatten()
        k_i = predicted_label

        w = np.zeros(input_shape, dtype=np.float32)
        r_tot = np.zeros(input_shape, dtype=np.float32)

        loop_i = 0
        # d = max(f_i[order[0]] - f_i[order[1]], 5)
        while k_i == label and loop_i < self.max_iter:

            pert = np.inf
            grad_orig = self.get_gradient(pert_image, k_i)
            for k in range(1, num_classes):
                grad_cur = self.get_gradient(pert_image, order[k])
                # set new w_k and new f_k
                w_k = grad_cur - grad_orig
                f_k = f_i[order[k]] - f_i[order[0]]

                pert_k = abs(f_k) / np.linalg.norm(w_k.flatten())

                # determine which w_k to use
                if pert_k < pert:
                    pert = pert_k
                    w = w_k

            # compute r_i and r_tot
            # Added 1e-4 for numerical stability
            r_i = pert * w / np.linalg.norm(w)
            r_tot = np.float32(r_tot + r_i)

            pert_image = np.clip(image + (1 + self.overshoot) * r_tot, self.min, self.max)

            r_tot = (pert_image - image) / (1 + self.overshoot)

            f_i = self.get_predict(pert_image)
            k_i = np.argmax(f_i)
            # dis = np.linalg.norm(self.model.autoencoder.predict([pert_image]) - pert_image)
            loop_i += 1

        r_tot = (1 + self.overshoot) * r_tot
        return r_tot, loop_i, pert_image, k_i

    def get_predict(self, image):
        bef = self.pred([image])[0]
        return bef[0]

    def get_gradient(self, image, label):
        if type(self.grads[label]) is type([]):
            dydx = K.gradients(self.before_softmax_tensor[..., label],
                               self.input_tensor)[0]
            
            # rand_x = self.model.get_encoder_classifier().get_layer('lambda_5').output
            self.grads[label] = K.function([self.input_tensor], [dydx])

        dydx = self.grads[label]([image])[0]
        return dydx


def gen_perturbation(self, img, source=5, target=6):
    self.perturb, self.loop_i, self.pert_image = self.deepfool(
        img, source, target)
    self.perturb = np.squeeze(self.perturb)
    return self.perturb, self.loop_i, self.pert_image
