# https://keras-cn-docs.readthedocs.io/zh_CN/latest/blog/autoencoder/
from keras.layers import GaussianNoise, concatenate, UpSampling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import multi_gpu_model

from data.DataGenerator import DataGenerator
from defences.autoencoder_abstract import Autoencoder_abstract, NoiseLayer
from utils import *


def conv_with_bn(x,channel, activation, name, stride=1, decoded=False, noise_layer=None,):
    x = Conv2D(channel, (3, 3), strides=stride, activation=None, 
            name=name + '_conv', padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization(name=name+'_bn')(x)
    if decoded:
        x = Activation(activation, name='decoded')(x)
    else:
        x = Activation(activation, name=name+'_ac')(x)
    if type(noise_layer) is dict and name+'_bn' in noise_layer and 'decoded' not in name:
        x = noise_layer[name+'_bn'](x)
    # elif noise_layer is not None:
    #     x = noise_layer(x)
    return x



class Unet(Autoencoder_abstract):
    def __init__(self, param, channels=3):
        super(Unet, self).__init__(param)
        print('build denoising unet bn')

        K.set_learning_phase(1)
        with open(param.get_conf('statistic_file'), 'r') as f:
            fsta = json.load(f)
        mus = fsta['dis_mus']
        stds = fsta['dis_stds']
        sta_layers = fsta['collected_neruals']
        alpha = 16
        noise_layers = {layers: NoiseLayer(mus=mus[i],stddevs=stds[i], alpha=alpha , 
                    name="au_blokc{}_noise".format(i)) for i, layers in enumerate(sta_layers)}
        self.input_shape = (
            self.param.get_conf()['train_image_size'], self.param.get_conf()['train_image_size'], channels,)
        input_img = Input(self.input_shape)
        noise_img = GaussianNoise(stddev=15 / 255.0, name='noise_input')(input_img)
        factor = 4
        conv1 = conv_with_bn(noise_img, name='au_block1_layer1', channel=64 // factor, activation='relu', noise_layer = noise_layers)
        conv1 = conv_with_bn(conv1, name='au_block1_layer2',  channel = 64 // factor, activation='relu', noise_layer = noise_layers)
        pass_1 = conv_with_bn(conv1, name='pass1', channel=4, activation='relu', noise_layer = noise_layers,)
        # pass_1 = conv1
        pool1 = MaxPooling2D(pool_size=(2, 2), name='au_blokc1_pool')(conv1)
        conv2 = conv_with_bn(pool1, name='au_block2_layer1', channel=128 // factor, activation='relu', noise_layer = noise_layers)
        conv2 = conv_with_bn(conv2, name='au_block2_layer2', channel=128 // factor, activation='relu', noise_layer = noise_layers)
        pass_2 = conv_with_bn(conv2, name='pass2', channel=8, activation='relu', noise_layer = noise_layers)
        # pass_2 = conv2
        pool2 = MaxPooling2D(pool_size=(2, 2), name='au_blokc2_pool')(conv2)
        conv3 = conv_with_bn(pool2, name='au_block3_layer1', channel=256 // factor, activation='relu', noise_layer = noise_layers)
        conv3 = conv_with_bn(conv3, name='au_block3_layer2', channel=256 // factor, activation='relu', noise_layer = noise_layers)
        pass_3 = conv_with_bn(conv3, name='pass3', channel=16, activation='relu', noise_layer = noise_layers)
        # pass_3 = conv3
        pool3 = MaxPooling2D(pool_size=(2, 2), name='au_blokc3_pool')(conv3)
        conv4 = conv_with_bn(pool3, name='au_block4_layer1', channel=512 // factor, activation='relu', noise_layer = noise_layers)
        conv4 = conv_with_bn(conv4, name='au_block4_layer2', channel=512 // factor, activation='relu', noise_layer = noise_layers)
        drop4 = Dropout(0.5, name='au_blokc4_dropout')(conv4)
        pass_4 = conv_with_bn(drop4, name='pass4', channel=32, activation='relu', noise_layer = noise_layers)
        # pass_4 = drop4
        pool4 = MaxPooling2D(pool_size=(2, 2), name='au_blokc4_pool')(drop4)

        conv5 = conv_with_bn(pool4, name='au_block5_layer1', channel=512 // factor, activation='relu', noise_layer = noise_layers)
        conv5 = conv_with_bn(conv5, name='au_block5_layer2', channel=512 // factor, activation='relu', noise_layer = noise_layers)
        drop5 = Dropout(0.5, name='au_blokc5_dropout')(conv5)

        up6 = UpSampling2D((2, 2), name='encoded')(drop5)

        up6 = conv_with_bn(up6, name='au_block6_layer1', channel=512 // factor, activation='relu', noise_layer = noise_layers)
        merge6 = concatenate([pass_4, up6], axis=3, name='au_blokc6_merge')
        conv6 = conv_with_bn(merge6, name='au_block6_layer2', channel=512 // factor, activation='relu', noise_layer = noise_layers)
        conv6 = conv_with_bn(conv6, name='au_block6_layer3', channel=512 // factor, activation='relu', noise_layer = noise_layers)

        up7 = UpSampling2D((2, 2), name='au_blokc7_upsample')(conv6)
        up7 = conv_with_bn(up7, name='au_block7_layer1', channel=256 // factor,  activation='relu', noise_layer = noise_layers)
        merge7 = concatenate([pass_3, up7], axis=3, name='au_blokc7_merge')
        conv7 = conv_with_bn(merge7, name='au_block7_layer2', channel=256 // factor, activation='relu', noise_layer = noise_layers,)
        conv7 = conv_with_bn(conv7, name='au_block7_layer3', channel=256 // factor, activation='relu', noise_layer = noise_layers,)

        up8 = UpSampling2D((2, 2), name='au_blokc8_upsample')(conv7)
        up8 = conv_with_bn(up8, name='au_block8_layer1', channel=128 // factor,  activation='relu', noise_layer = noise_layers)
        merge8 = concatenate([pass_2, up8], axis=3, name='au_blokc8_merge')
        conv8 = conv_with_bn(merge8, name='au_block8_layer2', channel=128 // factor, activation='relu', noise_layer = noise_layers)
        conv8 = conv_with_bn(conv8, name='au_block8_layer3', channel=128 // factor, activation='relu', noise_layer = noise_layers)

        up9 = UpSampling2D((2, 2), name='au_blokc9_upsample')(conv8)
        up9 = conv_with_bn(up9, name='au_block9_layer1', channel=64 // factor,  activation=None,)
        merge9 = concatenate([pass_1, up9], axis=3, name='au_blokc9_merge')
        conv9 = conv_with_bn(merge9, name='au_block9_layer2', channel=64 // factor, activation='relu', noise_layer = noise_layers,)
        conv9 = conv_with_bn(conv9, name='au_block9_layer3', channel=64 // factor, activation='relu', noise_layer = noise_layers,)
        decoded = conv_with_bn(conv9, activation='sigmoid', channel=channels, name='decoded', decoded=True)
        # decoded= AutoEncoderLoss()([input_img, decoded, noise_img])
        self.autoencoder = Model(inputs=input_img, outputs=decoded)
        # self.autoencoder.summary()
        return

        # if self.param.get_conf()['num_gpu'] > 1:
        #     self.autoencoder = multi_gpu_model(self.autoencoder, gpus=self.param.get_conf()['num_gpu'])

    # def train(self, x, y=None, train_epochs_rate = 1):
    #     K.set_learning_phase(1)
    #     if self.param.get_conf('num_gpu') > 1:
    #         parallel_model = multi_gpu_model(self.autoencoder, gpus=2)
    #         parallel_model.compile(loss='binary_crossentropy', optimizer='adam')
    #     else:
    #         self.autoencoder.compile(
    #             loss='binary_crossentropy',
    #             optimizer='adam'
    #         )

    #     batch_size = self.param.get_conf()['batch_size']

    #     data_exten = ImageDataGenerator(rotation_range=40,
    #                                     width_shift_range=0.2,
    #                                     height_shift_range=0.2,
    #                                     shear_range=0.2,
    #                                     zoom_range=0.2,
    #                                     horizontal_flip=True,
    #                                     fill_mode='nearest',
    #                                     zca_epsilon=1e-06,  # epsilon for ZCA whitening
    #                                     )

    #     def get_fit_encoder(x):
    #         data_gen = DataGenerator(x,
    #                                  self.param,
    #                                  batch_size=batch_size,
    #                                  preprocess=preprocess_autoencoder,
    #                                  transform=data_exten.random_transform,
    #                                  )
    #         len_gen = len(data_gen)
    #         while True:
    #             for idx, batch in enumerate(data_gen):
    #                 yield (batch, batch)
    #                 # print('batch.shape =',batch.shape)
    #                 if idx + 1 == len_gen:
    #                     data_gen.on_epoch_end()
    #                     break

    #     gen = get_fit_encoder(x)
    #     if self.param.get_conf('num_gpu') > 1:
    #         parallel_model.fit_generator(
    #             gen,
    #             epochs=self.param.get_conf('train_epoch_autoencoder') * train_epochs_rate,
    #             steps_per_epoch=np.ceil(len(x) / batch_size),
    #             shuffle=True,
    #             use_multiprocessing=True,
    #             workers=8)
    #     else:
    #         self.autoencoder.fit_generator(
    #             gen,
    #             epochs=self.param.get_conf('train_epoch_autoencoder') * train_epochs_rate,
    #             steps_per_epoch=np.ceil(len(x) / batch_size),
    #             shuffle=True,
    #             workers=1)

    def get_encoded_flatten(self):
        return self.encoded_imgs.reshape(len(self.encoded_imgs), 128 * 4 * 4)

    def dump_model(self, type_str = ''):
        name = '_'.join(['autoencoder_unet', self.param.get_conf()['model_prefix'], type_str,  get_date()]) + '.pkl'
        path = os.path.join(self.param.get_conf()['save_dir'], name)
        if not os.path.exists(self.param.get_conf()['save_dir']):
            os.makedirs(self.param.get_conf()['save_dir'])
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        print("autoencoder dump success at " + path)
        return path

    def dump_model_weights(self, type_str = ''):
        name = '_'.join(['autoencoder_unet_middle', self.param.get_conf()['model_prefix'], type_str,  get_date()]) + '.h5'
        path = os.path.join(self.param.get_conf()['save_dir'], name)
        if not os.path.exists(self.param.get_conf()['save_dir']):
            os.makedirs(self.param.get_conf()['save_dir'])
       
        self.autoencoder.save_weights(path)
        print("autoencoder weights dump success at " + path)
        return path

    def __setstate__(self, state):
        """
        Use to ensure `KerasClassifier` can be unpickled.

        :param state: State dictionary with instance parameters to restore.
        :type state: `dict`
        """
        self.__dict__.update(state)

        # Load and update all functionality related to Keras
        import os
        from keras.models import load_model

        full_path = os.path.join(self.param.get_conf()['save_dir'], state['model_name'])

        self.autoencoder = load_model(str(full_path),custom_objects={'NoiseLayer': NoiseLayer})
        self.autoencoder = Model(inputs = self.autoencoder.input, outputs = [self.autoencoder.get_layer("decoded").output])
        self._initialize_params()
