"""JSMA attack
"""
import logging
import copy
import math
import random
import numpy as np
import tensorflow as tf
from utils import *
from attacks.adversarial_attack import Adversarial
import math as m
from tqdm.gui import trange
from tqdm import tqdm
import urllib
from cleverhans.utils import other_classes
from cleverhans.utils_keras import KerasModelWrapper
from cleverhans.utils_tf import model_argmax
from cleverhans.attacks_tf import jacobian_graph, jacobian, saliency_map, apply_perturbations



def create_logger(name):
  """
  Create a logger object with the given name.

  If this is the first time that we call this method, then initialize the
  formatter.
  """
  base = logging.getLogger("reforcement")
  if len(base.handlers) == 0:
    ch = logging.StreamHandler(stream=sys.stdout)
    formatter = logging.Formatter('[%(levelname)s %(asctime)s %(name)s] ' +
                                  '%(message)s')
    ch.setFormatter(formatter)
    base.addHandler(ch)

  return base

np_dtype = np.dtype('float32')
tf_dtype = tf.as_dtype('float32')

_logger = create_logger("cleverhans.attacks.JSMA")

jsma_params = {
    'theta': 1.0,
    'gamma': 0.12,
    'clip_min': 0,
    'clip_max': 1
}

class SaliencyMapMethod(Adversarial):
    """
    The Jacobian-based Saliency Map Method (Papernot et al. 2016).
    Paper link: https://arxiv.org/pdf/1511.07528.pdf

    :param model: cleverhans.model.Model
    :param sess: optional tf.Session
    :param dtypestr: dtype of the data
    :param kwargs: passed through to super constructor

    :note: When not using symbolic implementation in `generate`, `sess` should
            be provided
    """

    def __init__(self, model, param, args=jsma_params, data_max=1, data_min=0):
        """
        Create a SaliencyMapMethod instance.
        Note: the model parameter should be an instance of the
        cleverhans.model.Model abstraction provided by CleverHans.
        """

        super(SaliencyMapMethod, self).__init__(model, param)

        self.structural_kwargs = [
            'theta', 'gamma', 'clip_max', 'clip_min'
        ]
        self.sess = K.get_session()
        fixed = dict((k, v) for k, v in args.items() if k in self.structural_kwargs)
        self.new_kwargs = dict(x for x in fixed.items())
        self.parse_params(**self.new_kwargs)

        self.param = param
        self.nb_classes = self.param.get_conf('classes_num')
        # self.model = KerasModelWrapper(Model([model.get_input_tensor()],[model.get_output_bef_softmax()]))
        self.model = KerasModelWrapper(model)
        self.input_placeholder = model.get_input_tensor()
        self.predictions = model.get_output_tensor()
        self.grads = jacobian_graph(self.predictions, self.input_placeholder, self.nb_classes)
        self.max = data_max
        self.min = data_min

    def attack(self, x_train, y_train, x_test, y_test, process=(None, None)):
        print('implement jsma attack')
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
                current_class = y[idx]
                target_classes = other_classes(self.nb_classes, current_class)
                target = random.choice(target_classes)
                pert_img, res, percent_perturb, new_label, loop_i = self.attack_batch(victim_img, target)
                
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
                     sample,
                     target,
                     feed=None):
        """
        TensorFlow implementation of the JSMA (see https://arxiv.org/abs/1511.07528
        for details about the algorithm design choices).
        :param sess: TF session
        :param x: the input placeholder
        :param predictions: the model's symbolic output (the attack expects the
                        probabilities, i.e., the output of the softmax, but will
                        also work with logits typically)
        :param grads: symbolic gradients
        :param sample: numpy array with sample input
        :param target: target class for sample input
        :param theta: delta for each feature adjustment
        :param gamma: a float between 0 - 1 indicating the maximum distortion
            percentage
        :param clip_min: minimum value for components of the example returned
        :param clip_max: maximum value for components of the example returned
        :return: an adversarial sample
        """

        # Copy the source sample and define the maximum number of features
        # (i.e. the maximum number of iterations) that we may perturb
        adv_x = copy.copy(sample)
        # count the number of features. For MNIST, 1x28x28 = 784; for
        # CIFAR, 3x32x32 = 3072; etc.
        nb_features = np.product(adv_x.shape[1:])
        # reshape sample for sake of standardization
        original_shape = adv_x.shape
        adv_x = np.reshape(adv_x, (1, nb_features))
        # compute maximum number of iterations
        max_iters = np.floor(nb_features * self.gamma / 2)

        # Find number of classes based on grads
        nb_classes = len(self.grads)

        increase = bool(self.theta > 0)

        # Compute our initial search domain. We optimize the initial search domain
        # by removing all features that are already at their maximum values (if
        # increasing input features---otherwise, at their minimum value).
        if increase:
            search_domain = {i for i in range(nb_features) if adv_x[0, i] < self.clip_max}
        else:
            search_domain = {i for i in range(nb_features) if adv_x[0, i] > self.clip_min}

        # Initialize the loop variables
        iteration = 0
        adv_x_original_shape = np.reshape(adv_x, original_shape)
        current = model_argmax(self.sess, self.input_placeholder, self.predictions, adv_x_original_shape, feed=feed)

        _logger.debug("Starting JSMA attack up to %s iterations", max_iters)
        # Repeat this main loop until we have achieved misclassification
        while (current != target and iteration < max_iters
                and len(search_domain) > 1):
            # Reshape the adversarial example
            adv_x_original_shape = np.reshape(adv_x, original_shape)

            # Compute the Jacobian components
            grads_target, grads_others = jacobian(
                self.sess,
                self.input_placeholder,
                self.grads,
                target,
                adv_x_original_shape,
                nb_features,
                nb_classes,
                feed=feed)

            if iteration % ((max_iters + 1) // 5) == 0 and iteration > 0:
                _logger.debug("Iteration %s of %s", iteration, int(max_iters))
            # Compute the saliency map for each of our target classes
            # and return the two best candidate features for perturbation
            i, j, search_domain = saliency_map(grads_target, grads_others,
                                            search_domain, increase)

            # Apply the perturbation to the two input features selected previously
            adv_x = apply_perturbations(i, j, adv_x, increase, self.theta, self.clip_min,
                                        self.clip_max)

            # Update our current prediction by querying the model
            current = model_argmax(self.sess, self.input_placeholder, self.predictions, adv_x_original_shape, feed=feed)

            # Update loop variables
            iteration = iteration + 1

        if current == target:
            _logger.debug("Attack succeeded using %s iterations", iteration)
        else:
            _logger.debug("Failed to find adversarial example after %s iterations",
                        iteration)

        # Compute the ratio of pixels perturbed by the algorithm
        percent_perturbed = float(iteration * 2) / nb_features

        # Report success when the adversarial example is misclassified in the
        # target class
        if current == target:
            return np.reshape(adv_x, original_shape), 1, percent_perturbed, current, iteration
        else:
            return np.reshape(adv_x, original_shape), 0, percent_perturbed, current, iteration

    def parse_params(self,
                    theta=1.,
                    gamma=1.,
                    clip_min=0.,
                    clip_max=1.):
        """
        Take in a dictionary of parameters and applies attack-specific checks
        before saving them as attributes.

        Attack-specific parameters:

        :param theta: (optional float) Perturbation introduced to modified
                    components (can be positive or negative)
        :param gamma: (optional float) Maximum percentage of perturbed features
        :param clip_min: (optional float) Minimum component value for clipping
        :param clip_max: (optional float) Maximum component value for clipping
        :param y_target: (optional) Target tensor if the attack is targeted
        """
        self.theta = theta
        self.gamma = gamma
        self.clip_min = clip_min
        self.clip_max = clip_max
