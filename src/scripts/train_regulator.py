import argparse

from attacks import *
from defences.autoencoder_abstract import AECallback, Autoencoder_abstract
# from attacks.CW_attack_model import CW_attack
from data import CifarData, SVHNData
from models import CnnModel
from utils import *

from utils import *
import random
import os


def main(args, param):
    if 'cifar' in param.get_conf('model_prefix'):
        Data = CifarData
        from defences.denoising_unet import Unet
    else:
        Data = SVHNData
        from defences.denoising_unet_bn import Unet
    data = Data(param)
    print("train new autoencoder")
    K.set_learning_phase(1)
    AE = Unet(param)
    # AE = Autoencoder_abstract.load_model(param.get_conf('autoencoder_path'))
    # AE.autoencoder.summary()
    if args.noise:
        print("loading weight form", param.get_conf('noise_autoencoder_weights'))
        AE.autoencoder.load_weights(param.get_conf('noise_autoencoder_weights'), 
                                    by_name=True, skip_mismatch=False)
    
    callback = []
    data_adv = Data(param, is_adversarial=1)
    if len(data_adv.x_test) >0 and len(data_adv.x_train)>0:
        if '.pkl' in param.get_conf('model_path') and os.path.exists(param.get_conf('model_path')):
            model = CnnModel.load_model(param.get_conf('model_path'))
            callback.append(AECallback(data, data_adv, AE, model, 
                        model_type=param.get_conf('model_prefix')))
    AE.train(data.x_train, callbacks=callback)
    save_path = AE.dump_model()
    param.set_conf('autoencoder_path', save_path)
    param.dump_conf()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str,
                        help='configure file.', default='cifar.json')
    parser.add_argument('--noise', '-n',
                        help='train noise autoencoder or unet', action='store_true')
    args = parser.parse_args()
    param = Param(args.config, prefix='../json')
    main(args, param)