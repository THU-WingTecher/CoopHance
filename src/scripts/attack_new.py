import argparse

from attacks import *
# from attacks.CW_attack_model import CW_attack
from data import CifarData, SVHNData
from models import CnnModel
from utils import *
import random
import os
from encoder_classifier import EncoderClassifier
from defences.autoencoder_abstract import Autoencoder_abstract
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

    if args.build is False:
        encoder_classifier = EncoderClassifier.load_model(
            param.get_conf("encoder_classifier_path"))
    else:
        assert 'pkl' in param.get_conf('autoencoder_path') and os.path.exists(
                param.get_conf('autoencoder_path'))
        AE = Autoencoder_abstract.load_model(param.get_conf('autoencoder_path'))

        assert 'pkl' in param.get_conf('model_path') and os.path.exists(
                param.get_conf('model_path'))
        model = CnnModel.load_model(param.get_conf('model_path'))
        

        encoder_classifier = EncoderClassifier(param, AE, model, channels=3)
        save_path = encoder_classifier.dump_model()
        param.set_conf("encoder_classifier_path", save_path)
        K.clear_session()
        encoder_classifier = EncoderClassifier.load_model(
            param.get_conf("encoder_classifier_path"))
        param.dump_conf()

    



    if args.attack:
        if 'deepfool' in param.get_conf('adversarial_type'):
            deepfool = Deepfool(encoder_classifier, param,data_max=1.0,data_min=0)
            deepfool.attack(data.x_train, data.y_train, data.x_test, data.y_test,
                            process=(preprocess_mnist, deprocess_mnist))
        elif 'fgsm' in param.get_conf('adversarial_type'):
            fgsm = FGSM(encoder_classifier,param,data_max=1.0,data_min=0)
            fgsm.attack(data.x_train, data.y_train, data.x_test, data.y_test,
                          process=(preprocess_mnist,deprocess_mnist),epsilon=5.0 /255)
        elif 'pgd' in param.get_conf('adversarial_type'):
            pgd = PGD(encoder_classifier,param,data_max=1.0,data_min=0,epsilon=1.0 /255, max_steps=50)
            pgd.attack(data.x_train, data.y_train, data.x_test, data.y_test,
                          process=(preprocess_mnist,deprocess_mnist))
        elif 'CW' in param.get_conf('adversarial_type'):
            CW = CarliniWagnerL2(encoder_classifier,param)
            CW.attack(data.x_train,data.y_train,data.x_test,data.y_test,process=(preprocess_mnist,deprocess_mnist))
        elif 'jsma' in param.get_conf('adversarial_type'):
            jsma = SaliencyMapMethod(encoder_classifier,param)
            jsma.attack(data.x_train, data.y_train, data.x_test, data.y_test,
                          process=(preprocess_mnist,deprocess_mnist))

    data_adversarial_au = Data(param, 1)

    data_adversarial = Data(param, 2)
    # model.batch_size=1
    assert len(data_adversarial_au.x_train)>0 and len(data_adversarial_au.x_test)>0,\
        'Please generate adversarial data for enhanced model first'
    encoder_classifier.evaluate(data_adversarial_au,
                                'adversarial for enhanced model',
                                vis=False)

    if len(data_adversarial.x_test) >0:
        encoder_classifier.evaluate(data_adversarial,
                                    "adversarial for classifier alone",
                                    vis=False)

    encoder_classifier.evaluate(data, "normmal data", vis=False)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c',
                        type=str,
                        help='configure file.',
                        default='cifar.json')
    parser.add_argument('--attack', '-a',
                        help='implement adversarial or not',
                        action='store_true')
    parser.add_argument('--build', '-b',
                        help='build new encoder classifier',
                        action='store_true')
    args = parser.parse_args()
    param = Param(args.config, prefix='../json')
    main(param, args)
