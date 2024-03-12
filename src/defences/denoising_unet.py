# https://keras-cn-docs.readthedocs.io/zh_CN/latest/blog/autoencoder/
from keras.layers import GaussianNoise, concatenate, Conv2DTranspose, noise
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import multi_gpu_model

from data.DataGenerator import DataGenerator
from .autoencoder_abstract import Autoencoder_abstract, NoiseLayer
from utils import *
from keras.optimizers import Adam

# from defences.autoencoder_abstract import OsvmCallback


class Unet(Autoencoder_abstract):
    def __init__(self, param, channels=3):
        super(Unet, self).__init__(param)
        print('build denoising unet')
        K.set_learning_phase(1)
        with open(param.get_conf('statistic_file'), 'r') as f:
            fsta = json.load(f)
        mus = fsta['dis_mus']
        stds = fsta['dis_stds']
        alphas = np.array([4, 4, 8, 8, 16, 16, 16, 16, 16]) 
        noise_layers = [NoiseLayer(mus=mus[i],stddevs=stds[i], alpha=alphas[i] , 
                    name="au_blokc{}_noise".format(i)) for i in range(len(mus))]
        self.input_shape = (
            self.param.get_conf()['train_image_size'], self.param.get_conf()['train_image_size'], channels,)
        input_img = Input(self.input_shape)
        noise_img = NoiseLayer(stddevs=10 / 255.0, name='noise_input')(input_img)
        factor = 4
        conv1 = Conv2D(64 // factor, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='au_blokc1_conv1')(noise_img)
        conv1 = Conv2D(64 // factor, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='au_blokc1_conv2')(conv1)
        conv1 = noise_layers[0](conv1)
        # pass_1 = Conv2D(1, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='au_block1_pass')(conv1)
        # pass_1 = conv1
        pool1 = MaxPooling2D(pool_size=(2, 2), name='au_blokc1_pool')(conv1)
        conv2 = Conv2D(128 // factor, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='au_blokc2_conv1')(pool1)
        conv2 = Conv2D(128 // factor, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='au_blokc2_conv2')(conv2)
        conv2 = noise_layers[1](conv2)
        pass_2 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='au_block2_pass')(conv2)
        # pass_2 = conv2
        pool2 = MaxPooling2D(pool_size=(2, 2), name='au_blokc2_pool')(conv2)
        conv3 = Conv2D(256 // factor, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='au_blokc3_conv1')(pool2)
        conv3 = Conv2D(256 // factor, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='au_blokc3_conv2')(conv3)
        conv3 = noise_layers[2](conv3)
        pass_3 = Conv2D(4, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='au_block3_pass')(conv3)
        # pass_3 = conv3
        pool3 = MaxPooling2D(pool_size=(2, 2), name='au_blokc3_pool')(conv3)
        conv4 = Conv2D(512 // factor, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='au_blokc4_conv1')(pool3)
        conv4 = Conv2D(512 // factor, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='au_blokc4_conv2')(conv4)
        drop4 = Dropout(0.5, name="au_block4_dropout")(conv4)
        drop4 = noise_layers[3](drop4)
        pass_4 = Conv2D(8, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='au_block4_pass')(drop4)
        # pass_4 = drop4
        pool4 = MaxPooling2D(pool_size=(2, 2), name='au_blokc4_pool')(drop4)

        conv5 = Conv2D(512 // factor, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='au_blokc5_conv1')(pool4)
        conv5 = Conv2D(512 // factor, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='au_blokc5_conv2')(conv5)
        drop5 = Dropout(0.5, name='au_blokc5_dropout')(conv5)
        drop5 = noise_layers[4](drop5)

        up6 = UpSampling2D((2, 2), name='encoded')(drop5)

        up6 = Conv2D(512 // factor, 2, activation='relu', padding='same', kernel_initializer='he_normal', name='au_blokc6_conv1')(up6)
        up6 = noise_layers[5](up6)
        merge6 = concatenate([pass_4, up6], axis=3, name='au_blokc6_merge')
        conv6 = Conv2D(512 // factor, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='au_blokc6_conv2')(merge6)
        conv6 = Conv2D(512 // factor, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='au_blokc6_conv3')(conv6)

        up7 = UpSampling2D((2, 2), name='au_blokc7_upsample')(conv6)
        up7 = Conv2D(256 // factor, 2, activation='relu', padding='same', kernel_initializer='he_normal', name='au_blokc7_conv1')(up7)
        up7 = noise_layers[6](up7)
        merge7 = concatenate([pass_3, up7], axis=3, name='au_blokc7_merge')
        conv7 = Conv2D(256 // factor, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='au_blokc7_conv2')(merge7)
        conv7 = Conv2D(256 // factor, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='au_blokc7_conv3')(conv7)
        
        up8 = UpSampling2D((2, 2), name='au_blokc8_upsample')(conv7)
        up8 = Conv2D(128 // factor, 2, activation='relu', padding='same', kernel_initializer='he_normal', name='au_blokc8_conv1')(up8)
        up8 = noise_layers[7](up8)
        merge8 = concatenate([pass_2, up8], axis=3, name='au_blokc8_merge')
        conv8 = Conv2D(128 // factor, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='au_blokc8_conv2')(merge8)
        conv8 = Conv2D(128 // factor, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='au_blokc8_conv3')(conv8)

        up9 = UpSampling2D((2, 2), name='au_blokc9_upsample')(conv8)
        up9 = Conv2D(64 // factor, 2, activation='relu', padding='same', kernel_initializer='he_normal', name='au_blokc9_conv1')(up9)
        up9 = noise_layers[8](up9)
        # merge9 = concatenate([pass_1, up9], axis=3, name='au_blokc9_merge')
        conv9 = Conv2D(64 // factor, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='au_blokc9_conv2')(up9)
        conv9 = Conv2D(64 // factor, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='au_blokc9_conv3')(conv9)
        decoded = Conv2D(3, 3, activation='sigmoid', padding='same', name='decoded')(conv9)
        # decoded= AutoEncoderLoss()([input_img, decoded, noise_img])
        self.autoencoder = Model(inputs=input_img, outputs=decoded)
        self.autoencoder.summary()
        return

        # if self.param.get_conf()['num_gpu'] > 1:
        #     self.autoencoder = multi_gpu_model(self.autoencoder, gpus=self.param.get_conf()['num_gpu'])

        # if self.param.get_conf()['num_gpu'] > 1:
        #     self.autoencoder = multi_gpu_model(self.autoencoder, gpus=self.param.get_conf()['num_gpu'])

    # def train(self, x, y=None, callbacks = None):
    #     K.set_learning_phase(1)
    #     if self.param.get_conf('num_gpu') > 1:
    #         parallel_model = multi_gpu_model(self.autoencoder, gpus=2)
    #         parallel_model.compile(loss='binary_crossentropy', optimizer= Adam(lr = 1e-4))
    #     else:
    #         self.autoencoder.compile(
    #             loss='binary_crossentropy',
    #             optimizer= Adam(lr = 1e-4)
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
    #                 # batch = batch + np.random.normal(loc=0.0, scale=2.0/255, size=(len(batch),)+self.input_shape)
    #                 yield (batch, batch)
    #                 # print('batch.shape =',batch.shape)
    #                 if idx + 1 == len_gen:
    #                     data_gen.on_epoch_end()
    #                     break

    #     gen = get_fit_encoder(x)
    #     if self.param.get_conf('num_gpu') > 1:
    #         parallel_model.fit_generator(
    #             gen,
    #             epochs=self.param.get_conf('train_epoch_autoencoder'),
    #             steps_per_epoch=np.ceil(len(x) / batch_size),
    #             shuffle=True,
    #             workers=8,
    #             callbacks=callbacks
    #             )
    #     else:
    #         self.autoencoder.fit_generator(
    #             gen,
    #             epochs=self.param.get_conf('train_epoch_autoencoder'),
    #             steps_per_epoch=np.ceil(len(x) / batch_size),
    #             shuffle=True,
    #             workers=2,
    #             callbacks=callbacks)

    def get_encoded_flatten(self):
        return self.encoded_imgs.reshape(len(self.encoded_imgs), 128 * 4 * 4)

    def dump_model(self):
        name = '_'.join(['autoencoder_unet', self.param.get_conf()['model_prefix'], get_date()]) + '.pkl'
        path = os.path.join(self.param.get_conf()['save_dir'], name)
        if not os.path.exists(self.param.get_conf()['save_dir']):
            os.makedirs(self.param.get_conf()['save_dir'])
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        print("autoencoder dump success at " + path)
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
        
