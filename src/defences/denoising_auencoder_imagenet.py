# https://keras-cn-docs.readthedocs.io/zh_CN/latest/blog/autoencoder/

from keras.preprocessing.image import ImageDataGenerator
from keras.layers import GaussianNoise
from data.DataGenerator import DataGenerator
from .encoder_abstract import EncoderAbstract
from utils import *
from keras.utils import multi_gpu_model

# from defences.encoder_abstract import OsvmCallback


class AutoEncoder(EncoderAbstract):
    def __init__(self, param, channels=3):
        super(AutoEncoder, self).__init__(param)
        print('build denoising autoencoder')
        self.input_shape = (self.param.get_conf()['train_image_size'], self.param.get_conf()['train_image_size'], channels,)
        input_img = Input(self.input_shape)
        noise_img = GaussianNoise(stddev=2 / 255.0)(input_img)
        factor1 = 4
        factor2 = 2
        factor3 = 2 
        factor4 = 4

        x = Conv2D(32 // factor1, (3, 3), activation='relu', name='au_block1_conv1', padding='same')(noise_img)
        x = Conv2D(32 // factor1, (3, 3), activation='relu', name='au_block1_conv2', padding='same')(x)
        x = MaxPooling2D(pool_size=(2, 2), padding='same', name='au_block1_pool1')(x)
        x = Conv2D(64 // factor2, (3, 3), activation='relu', name='au_block2_conv1', padding='same')(x)
        x = Conv2D(64 // factor2, (3, 3), activation='relu', name='au_block2_conv2', padding='same')(x)

        x = MaxPooling2D(pool_size=(2, 2), padding='same', name='au_block2_pool1')(x)
        x = Conv2D(128 // factor3, (3, 3), activation='relu', name='au_block3_conv1', padding='same')(x)
        x = Conv2D(128 // factor3, (3, 3), activation='relu', name='au_block3_conv2', padding='same')(x)
        x = Conv2D(128 // factor3, (3, 3), activation='relu', name='au_block3_conv3', padding='same')(x)
        # x = Conv2D(128 // factor3, (3, 3), activation='relu', name='au_block3_conv4', padding='same')(x)

        encoded = MaxPooling2D(pool_size=(2, 2), padding='same', name='encoded')(x)
        # print('encoded = ', encoded)

        x = Conv2D(128 // factor4, (3, 3), activation='relu', name='au_block4_conv1', padding='same')(encoded)
        x = Conv2D(128 // factor4, (3, 3), activation='relu', name='au_block4_conv2', padding='same')(x)
        # print('x = ', x)

        x = UpSampling2D((2, 2))(x)

        '''
        x = Conv2D(64 // factor, (3, 3), activation='relu', name='au_block5_conv1', padding='same')(x)
        x = Conv2D(64 // factor, (3, 3), activation='relu', name='au_block5_conv2', padding='same')(x)
        x = Conv2D(64 // factor, (3, 3), activation='relu', name='au_block5_conv3', padding='same')(x)
        '''
        x = Conv2D(128 // factor3, (3, 3), activation='relu', name='au_block5_conv1', padding='same')(x)
        x = Conv2D(128 // factor3, (3, 3), activation='relu', name='au_block5_conv2', padding='same')(x)
        x = Conv2D(128 // factor3, (3, 3), activation='relu', name='au_block5_conv3', padding='same')(x)
        # x = Conv2D(128 // factor3, (3, 3), activation='relu', name='au_block5_conv4', padding='same')(x)

        x = UpSampling2D((2, 2))(x)
        x = Conv2D(64 // factor2, (3, 3), activation='relu', name='au_block6_conv1', padding='same')(x)
        x = Conv2D(64 // factor2, (3, 3), activation='relu', name='au_block6_conv2', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(32 // factor1, (3, 3), activation='relu', name='au_block7_conv1', padding='same')(x)
        x = Conv2D(32 // factor1, (3, 3), activation='relu', name='au_block7_conv2', padding='same')(x)

        decoded = Conv2D(channels, (3, 3), activation='sigmoid', name='decoded', padding='same')(x)
        print('decoded = ', decoded)

        self.autoencoder = Model(inputs=input_img, outputs=decoded)

        # if self.param.get_conf()['num_gpu'] > 1:
        #     self.autoencoder = multi_gpu_model(self.autoencoder, gpus=self.param.get_conf()['num_gpu'])

    def train(self, x, y=None):
        K.set_learning_phase(1)
        if self.param.get_conf('num_gpu') >1:
            parallel_model = multi_gpu_model(self.autoencoder, gpus=2)
            parallel_model.compile(loss='binary_crossentropy',optimizer='adam')
        else:
            self.autoencoder.compile(
                loss='binary_crossentropy',
                optimizer='adam'
            )
        

        batch_size = self.param.get_conf()['batch_size']

        data_exten = ImageDataGenerator(preprocessing_function=preprocess_autoencoder,
                                        rotation_range=40,
                                        width_shift_range=0.2,
                                        height_shift_range=0.2,
                                        shear_range=0.2,
                                        zoom_range=0.2,
                                        horizontal_flip=True,
                                        fill_mode='nearest',
                                        zca_epsilon=1e-06,  # epsilon for ZCA whitening
                                        )

        def get_fit_encoder(x):
            data_gen = DataGenerator(x,
                                     self.param,
                                     batch_size=batch_size,
                                     preprocess=data_exten.standardize,
                                     )
            len_gen = len(data_gen)
            while True:
                for idx, batch in enumerate(data_gen):
                    # batch = batch + np.random.normal(loc=0.0, scale=2.0/255, size=(len(batch),)+self.input_shape) 
                    yield (batch, batch)
                    # print('batch.shape =',batch.shape)
                    if idx + 1 == len_gen:
                        data_gen.on_epoch_end()
                        break

        self.autoencoder.compile(loss='binary_crossentropy', optimizer='adam')
        gen = get_fit_encoder(x)
        if self.param.get_conf('num_gpu') >1:
            parallel_model.fit_generator(
            gen,
            epochs=self.param.get_conf('train_epoch_autoencoder'),
            steps_per_epoch=np.ceil(len(x) / batch_size),
            shuffle=True,
            workers=8)
        else:
            self.autoencoder.fit_generator(
                gen,
                epochs=self.param.get_conf('train_epoch_autoencoder'),
                steps_per_epoch=np.ceil(len(x) / batch_size),
                shuffle=True,
                workers=1)

    def get_encoded_flatten(self):
        return self.encoded_imgs.reshape(len(self.encoded_imgs), 128 * 4 * 4)
