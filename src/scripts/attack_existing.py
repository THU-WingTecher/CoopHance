import argparse

from attacks import *
# from attacks.CW_attack_model import CW_attack
from data import CifarData, SVHNData
from models import CnnModel
from utils import *
import random
import os

import tensorflow as tf
from defences.autoencoder_abstract import AECallback,Autoencoder_abstract
from defences.denoising_unet import Unet
from utils import *
import random
import os


def main(param, args):
    if 'cifar' in param.get_conf('model_prefix'):
        Data = CifarData
    else:
        Data = SVHNData
    data = Data(param)

    assert '.pkl' in param.get_conf('model_path') and os.path.exists(param.get_conf('model_path')), \
        'please give a trained classifier on \'model_path\' item of config file'
    model = CnnModel.load_model(param.get_conf('model_path'))

    assert '.pkl' in param.get_conf('autoencoder_path') and os.path.exists(param.get_conf('autoencoder_path')), \
        'please give a trained regulator on \'autoencoder_path\' item of config file'
    AE = Autoencoder_abstract.load_model(param.get_conf('autoencoder_path'))
    # AE.autoencoder.summary()
    if args.attack:
        if 'deepfool' in param.get_conf('adversarial_type'):
            deepfool = Deepfool(model, param,data_max=1.0,data_min=0)
            deepfool.attack(data.x_train, data.y_train, data.x_test, data.y_test,
                            process=(preprocess_mnist, deprocess_mnist))
        elif 'fgsm' in param.get_conf('adversarial_type'):
            fgsm = FGSM(model,param,data_max=1.0,data_min=0)
            fgsm.attack(data.x_train, data.y_train, data.x_test, data.y_test,
                          process=(preprocess_mnist,deprocess_mnist),epsilon=5.0 /255)
        elif 'pgd' in param.get_conf('adversarial_type'):
            pgd = PGD(model,param,data_max=1.0,data_min=0,epsilon=1.0 /255, max_steps=50)
            pgd.attack(data.x_train, data.y_train, data.x_test, data.y_test,
                          process=(preprocess_mnist,deprocess_mnist))
        elif 'CW' in param.get_conf('adversarial_type'):
            CW = CarliniWagnerL2(model,param)
            CW.attack(data.x_train,data.y_train,data.x_test,data.y_test,process=(preprocess_mnist,deprocess_mnist))
        elif 'jsma' in param.get_conf('adversarial_type'):
            jsma = SaliencyMapMethod(model,param)
            jsma.attack(data.x_train, data.y_train, data.x_test, data.y_test,
                          process=(preprocess_mnist,deprocess_mnist))

    data_adversarial = Data(param, 1)

    # model.batch_size=1

    model.evaluate(data_adversarial, 'adversarial without defence')
    model.evaluate(data_adversarial, 'adversarial', AE)
    model.evaluate(data, 'clean without defence')
    model.evaluate(data, 'clean', AE)

    AE.predict(data_adversarial.x_test, vis=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str,
                        help='configure file.', default='cifar.json')
    parser.add_argument('--attack', '-a',
                        help='implement attack', action='store_true')
    args = parser.parse_args()
    param = Param(args.config, prefix='../json')

    main(param, args)
