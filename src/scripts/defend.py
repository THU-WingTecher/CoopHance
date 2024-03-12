import argparse

from data import CifarData, SVHNData
from defences.denoising_unet import Unet
from models import CnnModel
from detector.inspector import Inspector
from utils import *
import random
import os
from defences.autoencoder_abstract import Autoencoder_abstract

def cal_adv(param, args):
    assert '.pkl' in param.get_conf('autoencoder_path') and os.path.exists(param.get_conf('autoencoder_path')), \
        'please train a regulator first'
    AE = Autoencoder_abstract.load_model(param.get_conf('autoencoder_path'))
  
    assert '.pkl' in param.get_conf('model_path') and os.path.exists(param.get_conf('model_path')), \
        'please train a classifier first'
    model = CnnModel.load_model(param.get_conf('model_path'))

    if 'cifar' in param.get_conf("model_prefix"):
        data_adv = CifarData(param, 1)
        data = CifarData(param)
    else:
        data_adv = SVHNData(param, 1)
        data = SVHNData(param)

    detector_enc = Inspector(param, AE, model, data, data_adv, detector_type='linear')

    if args.draw:
        TPs, FP_rate, aucs, both, blocker, enhancer = detector_enc.evaluate_ROC(
                            type_str = 'encoder_classifier')
        save_dir = os.path.join(param.get_conf('log_dir'), 'roc_data')
        check_dir(save_dir)
        save_name = '_'.join(['auc',param.get_conf('model_prefix'),
                    param.get_conf('adversarial_type'), args.config.split('.')[0], get_date()+'.npz'])
        np.savez(os.path.join(save_dir, save_name), both=both, blocker = blocker, enhancer=enhancer, TPs=TPs, FPs=FP_rate, aucs=aucs)

    if args.thres > 0:
        print('\n evaluate on threshold {:.4f}\n'.format(args.thres))
        TP, FP, both_det ,blocker_det, enhancer_det = detector_enc.evaluate(FP_rate=args.thres)
        print("evaluate, TP rate: {:.2f}%, FP rate: {:.2f}%".format(TP *100,FP*100))
        print("evaluate, block_only is {:.4f}, enchancer only is {:.4f}".format(blocker_det/len(data_adv.y_test), enhancer_det/len(data_adv.y_test)))

def get_thrs(dets, drop_rate):
    """
    Get filtering threshold by marking validation set.
    """
    thrs = dict()
    for name, detector in dets.items():
        num = int(len(detector.metric) * drop_rate)
        marks = detector.metric
        marks = np.sort(marks)
        thrs[name] = marks[-num] /255.0 / 255.0
    return thrs



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config','-c', type=str,
                        help='configure file.', default='../json/face.json')
    parser.add_argument('--thres', '-t', type=float, default=0.0,
                        help='give the FP rate and calcute TP rate')
    parser.add_argument('--draw', '-d', action='store_true',
                        help='choose to not draw the auc figure.')
   
    parser.add_argument('--cross_test', '-ct', type=str, default='')
    args = parser.parse_args()
    param = Param(args.config, prefix='../json')
    if args.cross_test != '':
        param.set_conf('cross_test', args.cross_test)
    K.clear_session()
    cal_adv(param, args)
