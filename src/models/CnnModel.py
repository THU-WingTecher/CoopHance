# -*- coding:utf-8 -*-
from abc import abstractclassmethod
from os import makedirs
from time import sleep
from keras.preprocessing.image import ImageDataGenerator

from models.keras_model import KerasClassifier
from utils import *
import abc



def lr_scheduler_adv(epoch):
    lr =0.1
    if epoch >= 100:
        lr /= 10
    if epoch >= 150:
        lr /= 100
    print('lr: {:.4f}'.format(lr))
    return lr

class CnnModel(metaclass=abc.ABCMeta):
    def __init__(self, param):
        self.param = param
        self.input_shape = (param.get_conf('train_image_size'), param.get_conf('train_image_size'), 3)
        # self.init_model()
        self.batch_size = param.get_conf('batch_size')
        self.classifier = None
        self.lr_scheduler_fun = None
    def set_learning_phase(self, learning_phase):
        K.set_learning_phase(learning_phase)

    @abc.abstractmethod
    def init_model(self):
        pass

    def train(self, data):
        # default
        # nb_epochs=20
        # batch_size=128
        from keras.preprocessing.image import ImageDataGenerator

        # if self.model_name == 'cnn':
        #     datagen = ImageDataGenerator(zoom_range=0.2, horizontal_flip=True)
        #     # zoom 0.2
        #     datagen = create_resnet_generator(dataset.x_train)
        #     callbacks_list = []
        #     batch_size = 128
        #     num_epochs = 200

     
        from keras.callbacks import LearningRateScheduler, ReduceLROnPlateau, EarlyStopping
        # zoom 0.2 horizontal_filp always True. change optimizer to sgd, and batch_size to 128. 
        datagen = ImageDataGenerator(rotation_range=15,
                                width_shift_range=5./32,
                                height_shift_range=5./32,
                                horizontal_flip = True,
                                zoom_range = 0.2,
                                preprocessing_function=preprocess_mnist)

        datagen.fit(data.x_train, seed=0)


        from keras.callbacks import LearningRateScheduler
        lr_scheduler = LearningRateScheduler(self.lr_scheduler_fun)
        callbacks_list = [lr_scheduler]
        batch_size = self.param.get_conf('batch_size')
        num_epochs = self.param.get_conf('train_epoch')

        # filepath="{}/models/original.hdf5".format(self.data_model)
        # checkpoint = ModelCheckpoint(filepath, monitor='val_acc', 
        #     verbose=1, save_best_only=True, mode='max')
        # callbacks_list.append(checkpoint)

        
        model_info = self.classifier.get_model().fit_generator(datagen.flow(data.x_train, 
            to_categorical(data.y_train), batch_size = batch_size),
            epochs = num_epochs,
            steps_per_epoch = data.x_train.shape[0] // batch_size,
            callbacks = callbacks_list, 
            validation_data = (data.x_test, to_categorical(data.y_test)), 
            # verbose = 2,
            use_multiprocessing=True,
            workers = 4)

    def train_adv(self, data, attack):


        from keras.callbacks import LearningRateScheduler
        batch_size = self.param.get_conf('batch_size')
        num_epochs = self.param.get_conf('train_epoch')
        filepath=os.path.join(self.param.get_conf('save_dir'), 'checkpoints')
        os.makedirs(filepath)
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=filepath,
        save_weights_only=True,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True)

        datagen = ImageDataGenerator(
                                # rotation_range=3,
                                # width_shift_range=2./32,
                                # height_shift_range=2./32,
                                # horizontal_flip = True,
                                # zoom_range = 0.1,
                                preprocessing_function=preprocess_mnist)

        model_info = self.classifier.get_model().fit_generator(
            datagen.flow(data.x_train, to_categorical(data.y_train,10), batch_size=self.batch_size),
            epochs = 20,
            steps_per_epoch = data.x_train.shape[0] // batch_size,
            # callbacks = callbacks_list, 
            validation_data = (data.x_test, to_categorical(data.y_test)), 
            # verbose = 2,
            # use_multiprocessing=True,
            workers = 4)


        np_epochs = self.param.get_conf()['train_epoch']
        def get_fit_encoder(x, y):
            flow = datagen.flow(x, to_categorical(y,10), batch_size=self.batch_size)
            # half = self.batch_size//2
            while True:
                if start:
                    sleep(1)
                    start=0
                batch, y = next(flow)
                batch = attack.attack_force(batch, y)
                yield (batch, y)

        gen = get_fit_encoder(data.x_train, data.y_train)


        from keras.callbacks import LearningRateScheduler
        lr_scheduler = LearningRateScheduler(lr_scheduler_adv)
        callbacks_list = [lr_scheduler, model_checkpoint_callback]
      

        
        model_info = self.classifier.get_model().fit_generator(gen,
            epochs = num_epochs,
            steps_per_epoch = data.x_train.shape[0] // batch_size,
            callbacks = callbacks_list, 
            validation_data = (data.x_test, to_categorical(data.y_test)), 
            # verbose = 2,
            # use_multiprocessing=True,
            workers = 1)

    def predict(self, x, y=None, type_str='', defence=None):
        # Evaluate the classifier on the test set
        test_preds = np.argmax(self.classifier.predict(x, defence=defence, batch_size=self.batch_size), axis=1)
        if y is not None:
             print("%s accuracy: %.2f%%" % (type_str, np.mean(test_preds==y) * 100))
        return test_preds
        # when result_dict is not empty, start record experiment results

    # to validate backdoor insert effectiveness
    # check whether the backdoor data with poison label is predicted by the model with poison label
    def evaluate(self, data, type_str='clean', defence=None):
        # Evaluate the classifier on the train set
        K.set_learning_phase(0)
        # Evaluate the classifier on the train set
        train_pred = self.predict(data.x_train, defence=defence)
        train_acc = np.mean(train_pred == data.y_train) 
        print("%s accuracy: %.2f%%" % (type_str + ' train', train_acc * 100))

        test_pred = self.predict(data.x_test ,defence=defence)
        test_acc = np.mean(test_pred == data.y_test) 
        print("%s accuracy: %.2f%%" % (type_str + ' test', test_acc * 100))
        '''
        # visualize predict
        for i in range(3):
            print(np.where(np.array(data.is_poison_test) == 1)[0][i])
            data.visiualize_img_by_idx(np.where(np.array(data.is_poison_test) == 1)[0][i], self.poison_preds[i], False)
        '''
        return train_pred, test_pred


    def predict_instance(self, x):
        return self.classifier.predict(x)[0]

    def get_input_shape(self):
        return self.input_shape

    def set_input_shape(self, input_shape):
        self.input_shape = input_shape

    def get_classifier(self):
        return self.classifier

    def set_classifier(self, classifier):
        self.classifier = classifier

    def get_input_tensor(self):
        return self.classifier.get_input_tensor()

    def get_output_tensor(self):
        return self.classifier.get_output_tensor()

    def get_output_bef_softmax(self):
        return self.get_classifier().get_output_bef_softmax()

    def dump_model(self, ):
        serialize_name = '_'.join([self.param.get_conf('model_prefix'), get_date()]) + '.pkl'
        if not os.path.isdir(self.param.get_conf()['save_dir']):
            os.makedirs(self.param.get_conf()['save_dir'], exist_ok=True)
        serialize_name = os.path.join(self.param.get_conf()['save_dir'], serialize_name)
        with open(serialize_name, 'wb') as f:
            pickle.dump(self, f)
        print('classifier dump success at', serialize_name)
        return serialize_name

    def get_model(self):
        return self.get_classifier().get_model()

    @staticmethod
    def load_model(model_path):
        K.set_learning_phase(0)
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print('load classifier success at', model_path)
        return model

    def collect_neurals(self, x, nerual_lists, defence = None):
        assert type(nerual_lists) is list
        res = [[] for n in nerual_lists]
        req_tensors = [self.classifier._model.get_layer(na).output for na in nerual_lists]
        # batch_size = self.param.get_conf()['batch_size']
        batch_size = 1
        fun = K.function([self.classifier._model.input],req_tensors)
        len_batch= int(np.ceil(len(x) / float(batch_size)))
        for batch_index in range(len_batch):
            begin, end = batch_index * batch_size, min((batch_index + 1) * batch_size, len(x))
            if type(x[0]) is str:
                raw_x = []
                # print('path = ', path)
                # print('self.data_path = ', self.data_path)
                for p in range(begin, end):
                    img = cv2.imread(x[p])[:, :, ::-1]
                    img = cv2.resize(img, self.train_shape)
                    raw_x.append(img)
                # import pdb
                # pdb.set_trace()

                pre_x = preprocess_autoencoder(np.array(raw_x))
            else:
                raw_x = x[begin:end]
                pre_x = preprocess_autoencoder(raw_x)
            if defence is not None:
                pre_x = defence(pre_x)
            neruals = fun([pre_x])
            for i in range(len(nerual_lists)):
                res[i].append(neruals[i])
        for i in range(len(nerual_lists)):
            res[i] = np.concatenate(res[i], axis=0)
        return res
