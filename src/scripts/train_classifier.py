import argparse

from attacks import *
# from attacks.CW_attack_model import CW_attack
from data import CifarData, SVHNData
from models import ResNet, CifarVGG, DenseNet
from utils import *


from utils import *
import random
import os


def main(args, param):
    if 'cifar' in param.get_conf('model_prefix'):
        if 'res' in param.get_conf('model_prefix'):
            Model = ResNet
        else:
            Model = CifarVGG
        Data = CifarData
    else:
        if 'res' in param.get_conf('model_prefix'):
            Model = ResNet
        else:
            Model = DenseNet
        Data = SVHNData
    data = Data(param)

    print('train new classifier')
    K.set_learning_phase(1)
    model = Model(param)
    model.train(data)
    model_path = model.dump_model()
    param.set_conf('model_path', model_path)
    param.dump_conf()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str,
                        help='configure file.', default='cifar.json')
    args = parser.parse_args()
    param = Param(args.config, prefix='../json')
    main(args, param)