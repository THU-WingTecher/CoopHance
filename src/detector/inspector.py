from matplotlib.pyplot import cla, psd
from numpy.core.records import format_parser
from numpy.core.shape_base import block
from numpy.lib.npyio import save
from utils import *
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.decomposition import PCA
import matplotlib.colors as mcolors
from thundersvm import OneClassSVM, SVC
# from sklearn.svm import OneClassSVM
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import time
class Inspector():
    def __init__(self, param, AE, classifier, data_clean, data_adv, detector_type='distribution') -> None:
        self.param = param
        self.AE = AE
        self.classifier = classifier
        self.thres = None
        self.FP_rate = None
        self.data_clean, self.data_adv = data_clean, data_adv
        self.detector_type = detector_type
        self.pred_adv = self.classifier.predict(data_adv.x_test, 
                                    type_str='', defence=AE)
        self.attack_success_index = np.where(self.pred_adv!=data_adv.y_test)[0]
       
        if detector_type is "linear":
          
            act_layer = self.param.get_conf('act_layer')
            
            use_detector = AE
            num_class = param.get_conf('classes_num')
            self.linear = []
        
            preds = classifier.predict(data_adv.x_train, 
                            defence=use_detector, type_str='')
           
            time_col = 0
            time_train = 0
            for cls in range(num_class):
             
                indx = np.where(data_clean.y_train==cls)[0][:2000]
                train_act = self.classifier.collect_neurals(data_clean.x_train[indx], 
                        [act_layer])[0].reshape((len(indx), -1))
                indx_poison = np.where(preds==cls)[0]
             
                st_col = time.time()
                poison_act = self.classifier.collect_neurals(data_adv.x_train[indx_poison],
                        [act_layer])[0].reshape((len(indx_poison), -1))
                time_col += time.time()-st_col
                train_act = np.concatenate((train_act,poison_act), axis=0)
                y = np.zeros(len(train_act), dtype=np.float32)
                y[len(indx): ] = 1
                frac = len(indx)/len(train_act)
                # frac=0.5
                self.linear.append(LogisticRegression(C=10,solver='liblinear', class_weight={0:1-frac, 1:frac}) )
                st_train = time.time()
                self.linear[cls].fit(train_act, y)
                time_train += time.time()-st_train
                pred = self.linear[cls].predict(train_act)
                print('train accuracy on class %d, poison is %.2lf, clean is %.2lf'\
                    %(cls, np.mean(pred[len(indx):]==y[len(indx):]), np.mean(pred[: len(indx)]==y[:len(indx)])))

                del train_act
                del y
                del indx
                del indx_poison
          
            self.recoder_adv = Detector_Linear(param, self.classifier, use_detector, self.linear, self.data_adv, act_layer)
            self.recoder_clean = Detector_Linear(param, self.classifier, use_detector, self.linear, self.data_clean, act_layer)
        else:
            assert False, 'not implement yet'
        self.len_data = len(data_adv.x_test)

    def evaluate_ROC(self, type_str = ''):
        self.get_thres(FP_rate=None)
        print('calculating auc value')
        TPs = []
        blocker = []
        enhancer = []
        both = []
        for idx in range(len(self.thres)):
            TP,_, both_det, blocker_det, enhancer_det = self.evaluate(self.thres[idx], self.FP_rate[idx])
            TPs.append(TP)
            both.append(both_det)
            enhancer.append(enhancer_det)
            blocker.append(blocker_det)
        TPs = np.array(TPs)
        both, enhancer, blocker = np.array(both) / self.len_data, \
                             np.array(enhancer) / self.len_data, np.array(blocker) / self.len_data
        aucs = auc(self.FP_rate, TPs)
        print("auc value on", type_str, "is", aucs)
        self.draw_roc(TPs, self.FP_rate, aucs, type_str)

        return TPs, self.FP_rate, aucs, both, blocker, enhancer
    
    def draw_roc(self, TPs, FPs, aucs, type_str=''):
        plt.style.use('classic')
        plt.plot(FPs, TPs, lw=1.5, label=type_str+" AUC=%.3f"%aucs)
        plt.xlabel("FPR",fontsize=15)
        plt.ylabel("TPR",fontsize=15)

        plt.title("ROC")
        plt.legend(loc="lower right")
        plt.show()
        save_dir = os.path.join(self.param.get_conf('log_dir'), 'roc')
        check_dir(save_dir)
        save_name = '_'.join(['auc',self.param.get_conf('model_prefix'),
                    self.param.get_conf('adversarial_type'), type_str, get_date()+'.jpg'])
        plt.savefig(os.path.join(save_dir, save_name))
    

    def evaluate(self, thre=None, FP_rate=None):
        assert not (thre is None and FP_rate is None)
        if FP_rate is None:
            FP_rate = self.get_FP(thre)
        if thre is None:
            self.get_thres(FP_rate)
            thre = self.thres
        index = np.arange(self.len_data)
        index = index[np.where(self.recoder_adv.metric <= thre)[0]]
        blocked = len(self.recoder_adv.metric ) - len(index)
        pre_res = self.pred_adv[index] == self.data_adv.y_test[index]
        ref_det = np.sum(self.pred_adv == self.data_adv.y_test)
        both = self.len_data - len(index) + np.sum(pre_res)
        TP_rate = (both) * 1.0 / self.len_data
        print("FP rate is {} and TP rate is {}, len index is {}, correct number is {}".format(FP_rate, TP_rate, len(index), np.sum(pre_res)))
        return TP_rate, FP_rate, both, blocked, ref_det
    
    def get_FP(self, thre):
        metric = np.sort(self.recoder_clean.metric)
        self.FP_rate = (self.len_data - np.searchsorted(metric, thre)) * 1.0 / self.len_data 
        return self.FP_rate

    def get_thres(self, FP_rate=None):
        if FP_rate is not None and type(FP_rate) is not np.ndarray:
            FP_rate = np.array(FP_rate)
        metric = np.sort(self.recoder_clean.metric)[::-1]
        if FP_rate is None:
            FP_rate = np.linspace(0, 0.9999999, 100)
        FP_number = (FP_rate * len(metric)).astype(np.int32)
        self.thres = metric[FP_number]
        self.FP_rate = FP_rate

    def load_adversarial(self, data_path):
        attacks = os.listdir(data_path)
        x_train = []
        y_train = []
        adversarial_idxs_train = []
        for attack in attacks:
            cur_attack_path = os.path.join(data_path, attack)
            train_path = os.path.join(cur_attack_path, 'train')
            # print(train_path, test_path)
            if os.path.exists(train_path):
                file_list = os.listdir(train_path)
                file_list.sort()
                for fi in file_list:
                    if '.npz' in fi:
                        train_adv_data = np.load(os.path.join(train_path, fi))
                        adversarial_idxs_train = train_adv_data['idx']
                        x_train = train_adv_data['imgs']
                        y_train = train_adv_data['labels']
               
                print('num of train adversarial image:',len(x_train))
        return np.array(x_train), np.array(y_train), np.array(adversarial_idxs_train)


class Detector_Linear():
    def __init__(self, param, classifier, detector, linear, data, neural_layer):
        len_data = len(data.x_test)
        acts = classifier.collect_neurals(data.x_test, [neural_layer])[0].reshape((len_data, -1))
        self.metric = np.zeros(len_data, dtype=np.float32)
        preds = classifier.predict(data.x_test, defence = detector)
        time_pre = 0
        for cls in range(param.get_conf('classes_num')):
            indx = np.where(preds==cls)[0]

            if len(indx)==0:
                continue

            self.metric[indx] = linear[cls].decision_function(acts[indx])

        return



