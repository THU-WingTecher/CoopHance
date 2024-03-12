import argparse

from attacks import Deepfool, FGSM, CarliniWagnerL2, SaliencyMapMethod, PGD
from data import SVHNData,CifarData
from defences.autoencoder_abstract import AECallback, Autoencoder_abstract
from models import ResNet, CifarVGG, DenseNet, CnnModel
from utils import *
from scipy import stats
import matplotlib.mlab as mlab


def cal_distribution(neurals, type_str):
    means = []
    stds = []
    res = []
    save_dir = os.path.join(param.get_conf('log_dir'), 'hist', type_str)
    check_dir(save_dir)
    for idx, neruals_layer in enumerate(neurals):
        
        print('calculating layer {}'.format(idx))
        # ner = ner.flatten()
        means_layers = []
        stds_layers = []
        res_layers = []
        # fig = plt.figure(figsize = (10,6))
        for ch, ner in enumerate(neruals_layer):
            ner = ner.flatten()
            mean = np.mean(ner)
            std = np.std(ner)
            res_layers += [stats.kstest(ner, 'norm', (mean, std))]
            means_layers.append(float(mean))
            stds_layers.append(float(std))

            # prob, bins, patches = plt.hist(ner, bins=100,alpha = 0.7, density = True)
            # y = stats.norm.pdf(bins, mean, std)
            # plt.plot(bins, y, 'r--')
            # plt.savefig(os.path.join(save_dir,
            #                 'layer{}-channel{}-{}'.format(idx, ch, get_date()) + '.jpg'))
            # plt.cla()
        means.append(means_layers)
        stds.append(stds_layers)
        res.append(res_layers)
    return means, stds, res
        
def caldulate_distances_ditribute(param, AE):
    
    with open(param.get_conf('statistic_file'), 'r') as f:
        fsta = json.load(f)
    
    if 'cifar' in param.get_conf('model_prefix'):
        Data = CifarData
    else:
        Data = SVHNData
    data = Data(param)
    # model = CnnModel.load_model(param.get_conf('model_path'))

    attack_methods = fsta['attack_methods']
    neural_list = fsta['collected_neruals']
    # neural_list=[]
    # for layer in AE.autoencoder.layers:
    #     if '_bn' in layer.name:
    #         neural_list.append(layer.name)

    distances = [[] for n in neural_list]
    seeds = fsta['attack_seeds']
    for i,attack in enumerate(attack_methods):
        print('\n\n------------------------------------------')
        print('evaluate on attack {}'.format(attack))
        param.set_conf('adversarial_type', attack)
        data = Data(param)
        # data.gen_shuffle_data(seeds=seeds[i])
        data_adversarial = Data(param, 1)
        assert len(data_adversarial.x_train) > 0 and len(data_adversarial.x_test), \
                'Please generate adversarial examples on {}'.format(attack) 
        get_remaind_data(data_adversarial.adversarial_idxs_train, 
            data_adversarial.adversarial_idxs_test, data)

        # model.evaluate(data_adversarial, 'adversarial without defence')
        # model.evaluate(data_adversarial, 'adversarial', AE)

        neurals_adv = AE.collect_neurals(data_adversarial.x_train, neural_list)
        neurals_norm = AE.collect_neurals(data.x_train, neural_list)
        for idx, (neu_a, neu_n) in enumerate(zip(neurals_adv, neurals_norm)):
            distances[idx].append((neu_a - neu_n)/(max(neu_a.max(),neu_n.max())-min(neu_a.min(),neu_n.min()))+1e-6)
    for idx in range(len(neural_list)):
        distances[idx] = np.concatenate(distances[idx], axis=0).transpose((3, 0, 1, 2))
    # npy_name = param.get_conf('model_prefix') + '_neurals-' + get_date() + '.npy'
    # save_path = os.path.join('../activations/', param.get_conf('model_prefix'), npy_name)
    # print('npy save name is', save_path)
    # np.save(save_path, distances)

    means, stds, res = cal_distribution(distances, param.get_conf('model_prefix')+'_distances')
    fsta['dis_stds'] = stds
    fsta['dis_mus'] = means
    fsta['dis_res_distribution'] = res
    # fsta['collected_neruals'] = neural_list
    with open(param.get_conf('statistic_file'),'w') as f:
        json.dump(fsta, f)

def get_sta_model(param):
    if 'cifar' in param.get_conf('model_prefix'):
        Data = CifarData
        from defences.denoising_unet_statistics import Unet   
    elif 'svhn' in param.get_conf('model_prefix'):
        Data = SVHNData
        from defences.denoising_unet_bn_statistics import Unet   
    data = Data(param)
    if '.pkl' in param.get_conf('autoencoder_path') and os.path.exists(param.get_conf('autoencoder_path')):
        AE = Autoencoder_abstract.load_model(param.get_conf('autoencoder_path'))
    else:
        print("train new autoencoder")
        AE = Unet(param)
        callback = []
        if '.pkl' in param.get_conf('model_path') and os.path.exists(param.get_conf('model_path')):
            model = CnnModel.load_model(param.get_conf('model_path'))
            data_adv = Data(param, is_adversarial=1)
            if len(data_adv.x_test)>0:
                callback.append(AECallback(data, data_adv, AE, model, 
                                    model_type=param.get_conf('model_prefix')))
        middle_rate=0.6
        AE.train(data.x_train, train_epochs_rate=middle_rate, callbacks = callback.copy())
        AE.dump_model_weights(type_str = 'middle')
        AE.train(data.x_train, train_epochs_rate=1, callbacks = callback, 
                    start_epoch=middle_rate*param.get_conf('train_epoch_autoencoder'))
        save_path = AE.dump_model()
        param.set_conf('autoencoder_path', save_path)
        param.dump_conf()
    return AE


def gen_statistic_ae(args, param):
    if args.attack =='none':
        return
    if 'cifar' in param.get_conf('model_prefix'):
        Data = CifarData
    else:
        Data = SVHNData
    data = Data(param)
    attacks = ['jsma', 'pgd', 'fgsm', 'deepfool', 'CW']
    assert '.pkl' in param.get_conf('model_path') and os.path.exists(param.get_conf('model_path')), \
                'please specify a classifier in \'model_path\' item of config file'
    model = CnnModel.load_model(param.get_conf('model_path'))
    data = Data(param)
    if args.attack!='all':
        attacks=[args.attack]
    for attack in attacks:
        print('\n\n------------------------------------------')
        print('craft adversarial examples with {}'.format(attack))
        param.set_conf('adversarial_type', attack)
        if 'deepfool' in attack:
            deepfool = Deepfool(model, param,data_max=1.0,data_min=0)
            deepfool.attack(data.x_train, data.y_train, data.x_test, data.y_test,
                            process=(preprocess_mnist, deprocess_mnist))
        elif 'fgsm' in attack:
            fgsm = FGSM(model,param,data_max=1.0,data_min=0)
            fgsm.attack(data.x_train, data.y_train, data.x_test, data.y_test,
                          process=(preprocess_mnist,deprocess_mnist),epsilon=5.0 /255)
        elif 'pgd' in attack:
            pgd = PGD(model,param,data_max=1.0,data_min=0,epsilon=1.0 /255, max_steps=50)
            pgd.attack(data.x_train, data.y_train, data.x_test, data.y_test,
                          process=(preprocess_mnist,deprocess_mnist))
        elif 'CW' in attack:
            CW = CarliniWagnerL2(model,param)
            CW.attack(data.x_train,data.y_train,data.x_test,data.y_test,process=(preprocess_mnist,deprocess_mnist))
        elif 'jsma' in attack:
            jsma = SaliencyMapMethod(model,param)
            jsma.attack(data.x_train, data.y_train, data.x_test, data.y_test,
                          process=(preprocess_mnist,deprocess_mnist))
    

def get_remaind_data(indexs_train, indexs_test, data):
    data.x_train = data.x_train[indexs_train]
    data.y_train = data.y_train[indexs_train]
    data.x_test = data.x_test[indexs_test]
    data.y_test = data.y_test[indexs_test]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str,
                        help='configure file.', default='cifar.json')
    parser.add_argument('--attack', '-a', type=str, choices=['none', 'all', 'pgd', 'fgsm', 'jsma', 'deepfool', 'CW'], default='none')
    args = parser.parse_args()
    param = Param(args.config, prefix='../json')
    gen_statistic_ae(args, param)
    autoencoder = get_sta_model(param)
    caldulate_distances_ditribute(param, autoencoder)
