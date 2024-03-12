# https://keras-cn-docs.readthedocs.io/zh_CN/latest/blog/autoencoder/
from keras.layers import GaussianNoise, concatenate, UpSampling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import multi_gpu_model

from data.DataGenerator import DataGenerator
from .autoencoder_abstract import Autoencoder_abstract, NoiseLayer
from utils import *


# from defences.autoencoder_abstract import OsvmCallback




class Unet(Autoencoder_abstract):
    def __init__(self, param, channels=3):
        super(Unet, self).__init__(param)
        print('build denoising unet')
        K.set_learning_phase(1)
        self.input_shape = (
            self.param.get_conf()['train_image_size'], self.param.get_conf()['train_image_size'], channels,)
        input_img = Input(self.input_shape)
        noise_img = GaussianNoise(stddev=15 / 255.0, name='noise_input')(input_img)
        factor = 4
        conv1 = Conv2D(64 // factor, 3, activation='relu', padding='same', 
                kernel_initializer='he_normal', name='au_blokc1_conv1')(noise_img)
        conv1 = Conv2D(64 // factor, 3, activation=None, padding='same', kernel_initializer='he_normal', name='au_blokc1_conv2')(conv1)
        conv1 = Activation('relu', name='au_block1_relu')(conv1)
        # pass_1 = Conv2D(1, 5, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
        # pass_1 = conv1
        pool1 = MaxPooling2D(pool_size=(2, 2), name='au_blokc1_pool')(conv1)
        conv2 = Conv2D(128 // factor, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='au_blokc2_conv1')(pool1)
        conv2 = Conv2D(128 // factor, 3, activation=None, padding='same', kernel_initializer='he_normal', name='au_blokc2_conv2')(conv2)
        conv2 = Activation('relu', name='au_block2_relu')(conv2)
        pass_2 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
        # pass_2 = conv2
        pool2 = MaxPooling2D(pool_size=(2, 2), name='au_blokc2_pool')(conv2)
        conv3 = Conv2D(256 // factor, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='au_blokc3_conv1')(pool2)
        conv3 = Conv2D(256 // factor, 3, activation=None, padding='same', kernel_initializer='he_normal', name='au_blokc3_conv2')(conv3)
        conv3 = Activation('relu', name='au_block3_relu')(conv3)
        pass_3 = Conv2D(4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
        # pass_3 = conv3
        pool3 = MaxPooling2D(pool_size=(2, 2), name='au_blokc3_pool')(conv3)
        conv4 = Conv2D(512 // factor, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='au_blokc4_conv1')(pool3)
        conv4 = Conv2D(512 // factor, 3, activation=None, padding='same', kernel_initializer='he_normal', name='au_blokc4_conv2')(conv4)
        conv4 = Activation('relu', name='au_block4_relu')(conv4)
        drop4 = Dropout(0.5, name='au_blokc4_dropout')(conv4)
        pass_4 = Conv2D(8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(drop4)
        # pass_4 = drop4
        pool4 = MaxPooling2D(pool_size=(2, 2), name='au_blokc4_pool')(drop4)

        conv5 = Conv2D(512 // factor, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='au_blokc5_conv1')(pool4)
        conv5 = Conv2D(512 // factor, 3, activation=None, padding='same', kernel_initializer='he_normal', name='au_blokc5_conv2')(conv5)
        conv5 = Activation('relu', name='au_block5_relu')(conv5)
        drop5 = Dropout(0.5, name='au_blokc5_dropout')(conv5)

        up6 = UpSampling2D((2, 2), name='encoded')(drop5)

        up6 = Conv2D(512 // factor, 2, activation=None, padding='same', kernel_initializer='he_normal', name='au_blokc6_conv1')(up6)
        up6 = Activation('relu', name='au_block6_relu')(up6)
        merge6 = concatenate([pass_4, up6], axis=3, name='au_blokc6_merge')
        conv6 = Conv2D(512 // factor, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='au_blokc6_conv2')(merge6)
        conv6 = Conv2D(512 // factor, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='au_blokc6_conv3')(conv6)

        up7 = UpSampling2D((2, 2), name='au_blokc7_upsample')(conv6)
        up7 = Conv2D(256 // factor, 2, activation=None, padding='same', kernel_initializer='he_normal', name='au_blokc7_conv1')(up7)
        up7 = Activation('relu', name='au_block7_relu')(up7)
        merge7 = concatenate([pass_3, up7], axis=3, name='au_blokc7_merge')
        conv7 = Conv2D(256 // factor, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='au_blokc7_conv2')(merge7)
        conv7 = Conv2D(256 // factor, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='au_blokc7_conv3')(conv7)

        up8 = UpSampling2D((2, 2), name='au_blokc8_upsample')(conv7)
        up8 = Conv2D(128 // factor, 2, activation=None, padding='same', kernel_initializer='he_normal', name='au_blokc8_conv1')(up8)
        up8 = Activation('relu', name='au_block8_relu')(up8)
        merge8 = concatenate([pass_2, up8], axis=3, name='au_blokc8_merge')
        conv8 = Conv2D(128 // factor, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='au_blokc8_conv2')(merge8)
        conv8 = Conv2D(128 // factor, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='au_blokc8_conv3')(conv8)

        up9 = UpSampling2D((2, 2), name='au_blokc9_upsample')(conv8)
        up9 = Conv2D(64 // factor, 2, activation=None, padding='same', kernel_initializer='he_normal', name='au_blokc9_conv1')(up9)
        up9 = Activation('relu', name='au_block9_relu')(up9)
        # merge9 = concatenate([pass_1, up9], axis=3, name='au_blokc9_merge')
        conv9 = Conv2D(64 // factor, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='au_blokc9_conv2')(up9)
        conv9 = Conv2D(64 // factor, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='au_blokc9_conv3')(conv9)
        decoded = Conv2D(3, 3, activation='sigmoid', padding='same', name='decoded')(conv9)
        # decoded= AutoEncoderLoss()([input_img, decoded, noise_img])
        self.autoencoder = Model(inputs=input_img, outputs=decoded)

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
        name = '_'.join(['autoencoder_unet_sta', self.param.get_conf()['model_prefix'], type_str,  get_date()]) + '.pkl'
        path = os.path.join(self.param.get_conf()['save_dir'], name)
        if not os.path.exists(self.param.get_conf()['save_dir']):
            os.makedirs(self.param.get_conf()['save_dir'])
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        print("autoencoder dump success at " + path)
        return path

    def dump_model_weights(self, type_str = ''):
        name = '_'.join(['autoencoder_unet_sta', self.param.get_conf()['model_prefix'], type_str,  get_date()]) + '.h5'
        path = os.path.join(self.param.get_conf()['save_dir'], name)
        if not os.path.exists(self.param.get_conf()['save_dir']):
            os.makedirs(self.param.get_conf()['save_dir'])
       
        self.autoencoder.save_weights(path)
        print("autoencoder weights dump success at " + path)
        return path
