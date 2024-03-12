# https://keras-cn-docs.readthedocs.io/zh_CN/latest/blog/autoencoder/
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import multi_gpu_model
from models import CifarVGG, ResNet, DenseNet
from models.ResNet18 import ResNet18
from models.DenseNet import densenet
from data.DataGenerator import DataGenerator
from encoder_classifier.encoder_abstract import EncoderClassifier_abstract
from utils import *
# from defences.autoencoder_abstract import OsvmCallback


class EncoderClassifier(EncoderClassifier_abstract):
    def __init__(self, param, autoencoder=None, classifier=None, channels=1):
        super(EncoderClassifier, self).__init__(param)
        K.set_learning_phase(1)
        if autoencoder is None:
            # load poison model
            model_path = self.param.get_conf()['autoencoder_path']
            with open(model_path, 'rb') as f:
                autoencoder_ = pickle.load(f)
        else:
            autoencoder_ = autoencoder
        autoencoder = autoencoder_.get_autoencoder()
        input_img = autoencoder_.get_input_tensor()
        decoded = autoencoder_.get_decoded()
        # x = input_img
        # for layer in autoencoder.layers:
            
        #     if 'input' not in layer.name:
        #         x = layer(x)
        #     if 'decoded' in layer.name:
        #         decoded = x
        #     if "encoded" in layer.name:
        #         encoded = x
        
        self.autoencoder = Model(inputs=input_img, outputs=decoded)
        # del autoencoder
        # if param.get_conf('model_prefix') == 'cifar':
        #     x = Lambda(lambda img_input: img_input*255.0)(decoded)
        # else:
        def round_decoded(x):
            return K.round(x*255)/255.0
        # x = Lambda(function=round_decoded,name = "round")(decoded)
        x = decoded
        
        classifier_ = classifier
        if isinstance(classifier, CifarVGG):
            classifier = classifier_.get_classifier().get_model()
            for layer in classifier.layers:
                #if 'dense' in layer.name or 'prediction' in layer.name:
                layer.trainable = False
                if "input" not in layer.name:
                    x = layer(x)
            preds_ = x
            del classifier
            self.model = Model(inputs=input_img, outputs=[decoded, preds_])
        else:
            if isinstance(classifier, ResNet):
                _, preds_ = ResNet18(classes = 10,input_shape=None, input_tensor = decoded)
            elif isinstance(classifier, DenseNet):
                _, preds_ = densenet(num_classes = 10, input_shape=None, input_tensor = decoded)
            extra=0
            layers_cla =  classifier = classifier_.get_classifier().get_model().layers
            self.model = Model(inputs=input_img, outputs=[decoded, preds_])
            # for idx, la1 in enumerate(self.model.layers):
            #     if 'au_' in la1.name or 'decoded' in la1.name or 'encoded' in la1.name or 'input' in la1.name:
            #         extra+=1
            #         continue
            #     # 
            # new_c = self.model.layers
            # for idx, la2 in enumerate(layers_cla[1:]):
            #     if new_c[idx+extra].name != la2.name:
            #         print(new_c[idx+extra].name, la2.name)
            #         raise('error')
            #     new_c[idx+extra].set_weights(la2.get_weights())
            self.model.load_weights(os.path.join(param.get_conf('save_dir'),
                    classifier_.get_classifier().model_name),by_name=True, skip_mismatch=True)
        self.decoded_imgs = None

    def dump_model(self):
        if "rand" in self.param.get_conf("autoencoder_path"):
            name = '_'.join(
            ['encoder_concat_rand',
             self.param.get_conf()['model_prefix'],
             get_date()]) + '.pkl'
        if "unet" in self.param.get_conf("autoencoder_path"):
            name = '_'.join(
            ['encoder_concat_unet',
             self.param.get_conf()['model_prefix'],
             get_date()]) + '.pkl'
        elif 'noise' in self.param.get_conf("autoencoder_path"):
            name = '_'.join(
            ['encoder_concat_noise',
             self.param.get_conf()['model_prefix'],
             get_date()]) + '.pkl'
        else:
            name = '_'.join(
                ['encoder_concat',
                self.param.get_conf()['model_prefix'],
                get_date()]) + '.pkl'
        path = os.path.join(self.param.get_conf()['save_dir'], name)
        if not os.path.exists(self.param.get_conf()['save_dir']):
            os.makedirs(self.param.get_conf()['save_dir'])
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        print("encoder classifier dump success at " + path)
        return path

    def train(self, data, is_restore=False):
        pass
        # if self.param.get_conf('num_gpu') > 1:
        #     autoencoder_multi = multi_gpu_model(
        #         self.autoencoder, gpus=self.param.get_conf('num_gpu'))

        #     model_multi = multi_gpu_model(self.model,
        #                                   gpus=self.param.get_conf('num_gpu'))

        #     autoencoder_multi.compile(loss='binary_crossentropy',
        #                               optimizer='adadelta')

        #     model_multi.compile(loss={
        #         'decoded': 'binary_crossentropy',
        #         'softmax': 'categorical_crossentropy'
        #     },
        #                         loss_weights={
        #                             'decoded': 1,
        #                             'softmax': 0.001
        #                         },
        #                         optimizer='adam',
        #                         metrics=['accuracy'])

        # else:
        #     self.autoencoder.compile(loss='binary_crossentropy',
        #                              optimizer='adadelta')
        #     self.model.compile(loss={
        #         'decoded': 'binary_crossentropy',
        #         'softmax': 'categorical_crossentropy'
        #     },
        #                        loss_weights={
        #                            'decoded': 1,
        #                            'softmax': 0.001
        #                        },
        #                        optimizer='adam',
        #                        metrics=['accuracy'])

        # x = data.x_train
        # y = data.y_train
        # batch_size = self.param.get_conf()['batch_size']

        # # data extension
        # data_exten = ImageDataGenerator(
        #     preprocessing_function=preprocess_autoencoder,
        #     rotation_range=40,
        #     width_shift_range=0.2,
        #     height_shift_range=0.2,
        #     shear_range=0.2,
        #     zoom_range=0.2,
        #     horizontal_flip=True,
        #     fill_mode='nearest',
        #     zca_epsilon=1e-06,  # epsilon for ZCA whitening
        # )

        # #data generator for aotoencoder
        # def gen_fit_encoder(x):
        #     data_gen = DataGenerator(
        #         x,
        #         self.param,
        #         batch_size=batch_size,
        #         preprocess=data_exten.standardize,
        #     )
        #     len_gen = len(data_gen)
        #     while True:
        #         for idx, batch in enumerate(data_gen):
        #             # batch = batch + np.random.normal(loc=0.0, scale=2.0/255, size=(len(batch),)+self.input_shape)
        #             yield (batch, batch)
        #             # print('batch.shape =',batch.shape)
        #             if idx + 1 == len_gen:
        #                 data_gen.on_epoch_end()
        #                 break

        # if not is_restore:

        #     gen = gen_fit_encoder(x)
        #     if self.param.get_conf()['num_gpu'] > 1:
        #         autoencoder_multi.fit_generator(
        #             gen,
        #             epochs=self.param.get_conf('train_epoch_aueocndoer'),
        #             steps_per_epoch=x.shape[0] / batch_size,
        #             shuffle=True,
        #             workers=8)
        #     else:
        #         self.autoencoder.fit_generator(
        #             gen,
        #             epochs=self.param.get_conf('train_epoch_autoencoder'),
        #             steps_per_epoch=x.shape[0] / batch_size,
        #             shuffle=True,
        #             workers=1)

        # #data generator for encoder classifier
        # def gen_fit(x, y):
        #     data_gen = DataGenerator(
        #         x,
        #         self.param,
        #         y,
        #         batch_size=batch_size,
        #         preprocess=data_exten.standardize,
        #     )
        #     len_gen = len(data_gen)
        #     while True:
        #         for idx, batch in enumerate(data_gen):
        #             # batch = batch + np.random.normal(loc=0.0, scale=2.0/255, size=(len(batch),)+self.input_shape)
        #             yield (batch[0], {
        #                 'decoded': batch[0],
        #                 'softmax': batch[1]
        #             })
        #             # print('batch.shape =',batch.shape)
        #             if idx + 1 == len_gen:
        #                 data_gen.on_epoch_end()
        #                 break

        # gen = gen_fit(x, y)

        # if self.param.get_conf('num_gpu') > 1:
        #     model_multi.fit_generator(
        #         gen,
        #         epochs=self.param.get_conf('train_epoch'),
        #         steps_per_epoch=np.ceil(len(x) / batch_size),
        #         shuffle=True,
        #         workers=8)
        # else:
        #     self.model.fit_generator(gen,
        #                              epochs=self.param.get_conf('train_epoch'),
        #                              steps_per_epoch=np.ceil(
        #                                  len(x) / batch_size),
        #                              shuffle=True,
        #                              workers=1)
