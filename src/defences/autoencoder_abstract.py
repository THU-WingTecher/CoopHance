# - * - coding: utf - 8 -
# *-
from keras.engine import Layer
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.training_utils import multi_gpu_model
import tensorflow as tf
from keras.layers.merge import dot
# from keras.layers import DepthwiseConv2D
from keras.losses import binary_crossentropy,mean_squared_error
from tqdm import trange
from keras import regularizers
from utils import *
import keras.initializers

class AECallback(Callback):
    def __init__(self, data_clean, data_adv, AE, classifier, print_epoch=5, model_type='mnist'):
        super(AECallback, self).__init__()

        self.AE = AE
        self.model_type = model_type
        self.print_epoch = print_epoch
        self.x_clean = data_clean.x_test
        self.y_clean = data_clean.y_test
        self.x_adv = data_adv.x_test
        self.y_adv = data_adv.y_test
        self.classifier = classifier
        # self.osvm = OneClassSVM(nu=0.05, gamma='auto', kernel="rbf")
    def on_epoch_end(self, epoch, logs):
        if epoch % self.print_epoch == 0:
            print('\n--------test---------------')
            refine_clean = self.AE.predict(self.x_clean)
            self.classifier.predict(refine_clean, self.y_clean, type_str='clean refine data')
            if len(self.x_adv)>0:
                refine_adv = self.AE.predict(self.x_adv)
                self.classifier.predict(refine_adv, self.y_adv, type_str='poison refine data')
            # print("loss of clean test is {:.4f}, loss of adv test is {:.4f}".format(
            #     cross_entropy_error(preprocess_autoencoder(refine_clean), preprocess_autoencoder(self.x_clean)),
            #     cross_entropy_error(preprocess_autoencoder(refine_adv), preprocess_autoencoder(self.x_adv))))
            print('---------------------------')


class Autoencoder_abstract():
    def __init__(self, param):
        self.autoencoder = None
        self.encoded_imgs = None
        self.decoded_imgs = None
        self.param = param
        # self.lr_decay = self.param.get_conf("lr_scheduler")

    def __getstate__(self):
        """
        Use to ensure `KerasClassifier` can be pickled.

        :return: State dictionary with instance parameters.
        :rtype: `dict`
        """

        state = self.__dict__.copy()

        # Remove the unpicklable entries

        del state['autoencoder']
        del state['encoded_imgs']
        del state['decoded_imgs']
        model_name = '_'.join(['autoencoder', get_date()]) + '.h5'
        state['model_name'] = model_name
        try:
            del state["autoencoder_train"]
            self.save(model_name, path=self.param.get_conf()['save_dir'],is_train=True)
        except:
            self.save(model_name, path=self.param.get_conf()['save_dir'])

        # model_name = str(time.time()) + '.h5'
        
        
        return state

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
        
        try:
            self.autoencoder = load_model(str(full_path),custom_objects={'AutoEncoderDiffLoss': AutoEncoderDiffLoss})
            loss_layer = self.autoencoder.get_layer('diff_loss')
            self.autoencoder_train = Model(inputs=self.autoencoder.input, outputs = [loss_layer.output])
            self.autoencoder = Model(inputs = self.autoencoder.input, outputs = [self.autoencoder.get_layer("decoded").output])
        except:
            self.autoencoder = load_model(str(full_path))
        self._initialize_params()
        

        # self._model = self.model
        # if self.param.get_conf()['num_gpu'] > 1:
        #     self.model = multi_gpu_model(self.model, gpus=self.param.get_conf()['num_gpu'])
        # self._initialize_params()

    def dump_model(self):
        name = '_'.join(['autoencoder', self.param.get_conf()['model_prefix'], get_date()]) + '.pkl'
        path = os.path.join(self.param.get_conf()['save_dir'], name)
        if not os.path.exists(self.param.get_conf()['save_dir']):
            os.makedirs(self.param.get_conf()['save_dir'])
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        print("autoencoder dump success at " + path)
        return path

    def save(self, filename, path=None, is_train = False):
        """
        Save a model to file in the format specific to the backend framework. For Keras, .h5 format is used.

        :param filename: Name of the file where to store the model.
        :type filename: `str`
        :param path: Path of the folder where to store the model. If no path is specified, the model will be stored in
                     the default data location of the library `DATA_PATH`.
        :type path: `str`
        :return: None
        """
        import os

        if path is None:
            from conf import DATA_PATH
            full_path = os.path.join(DATA_PATH, filename)
        else:
            full_path = os.path.join(path, filename)
        folder = os.path.split(full_path)[0]
        if not os.path.exists(folder):
            os.makedirs(folder)
        if is_train:
            self.autoencoder_train.save(str(full_path))
        else:
            self.autoencoder.save(str(full_path))
        logger.info('Model saved in path: %s.', full_path)

    def _initialize_params(self):
        self.encoded_imgs = None
        self.decoded_imgs = None

    @staticmethod
    def load_model(path):
        K.set_learning_phase(0)
        with open(path, 'rb') as f:
            model = pickle.load(f)
            print('load autoencoder success at ' + path)
        return model

    def predict(self, x, vis=False):
        '''
        print('layer name in self.model.layers')
        for layer in self.model.layers:
            print(layer.name)

        print('\n')

        print('layer name in self._model.layers')
        for layer in self._model.layers:
            print(layer.name)

        print('\n')
        '''
        # for layer in self._model:
        #     print(layer.name)
        decoded_imgs = np.zeros_like(x).astype(np.float32)
        batch_size = self.param.get_conf()['batch_size']
        vis_img = None
        for batch_index in range(int(np.ceil(len(x) / float(batch_size)))):
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
            decoded = self.autoencoder.predict(pre_x) * 255.0
            decoded_imgs[begin:end] = decoded
            if vis_img is None:
                vis_img = raw_x
        decoded_imgs = np.squeeze(np.array(decoded_imgs))
        if vis:
            self.visualize_autoencoder(vis_img, decoded_imgs)
        return decoded_imgs

    def collect_neurals(self, x, nerual_lists):
        assert type(nerual_lists) is list
        res = [[] for n in nerual_lists]
        req_tensors = [self.autoencoder.get_layer(na).output for na in nerual_lists]
        batch_size = self.param.get_conf()['batch_size']
        fun = K.function([self.autoencoder.input],req_tensors)
        for batch_index in range(int(np.ceil(len(x) / float(batch_size)))):
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
            neruals = fun([pre_x])
            for i in range(len(nerual_lists)):
                res[i].append(neruals[i])
        for i in range(len(nerual_lists)):
            res[i] = np.concatenate(res[i], axis=0)
        return res

    def visualize_autoencoder(self, x_test, decoded_imgs):
        n = 10
        x_test = x_test[:n + 1].astype(np.uint8)
        x_decoded = decoded_imgs[:n + 1]
        x_decoded = x_decoded.astype(np.uint8)
        plt.figure(figsize=(20, 4))
        check_dir(os.path.join(self.param.get_conf('log_dir'), 'decoded'))
        for i in range(1, n + 1):
            # display original
            ax = plt.subplot(2, n, i)
            if x_test.shape[3] == 1:
                plt.imshow(np.squeeze(x_test[i]), cmap='gray')
            else:
                plt.imshow(x_test[i])
            # plt.gray()
            # ax.get_xaxis().set_visible(False)
            # ax.get_yaxis().set_visible(False)

            # display reconstruction
            ax = plt.subplot(2, n, i + n)
            if x_test.shape[3] == 1:
                plt.imshow(np.squeeze(x_decoded[i]), cmap='gray')
            else:
                plt.imshow(x_decoded[i])
            # plt.gray()
            # ax.get_xaxis().set_visible(False)
            # ax.get_yaxis().set_visible(False)
        plt.savefig(os.path.join(self.param.get_conf('log_dir'), 'decoded', 'decoded_' + get_date() + '.jpg'))

        try:
            plt.show()
        except:
            print('can visualize')

    def visualize_encoder(self):
        n = 10
        try:
            plt.figure(figsize=(20, 16))
            for i in range(1, n + 1):
                ax = plt.subplot(1, n, i)
                # print('label = ', self.y_test[i])

                plt.imshow(self.encoded_imgs[i][:, :, :32].reshape(4, 4 * 32).T)
                # plt.gray()
                # ax.get_xaxis().set_visible(False)
                # ax.get_yaxis().set_visible(False)
            plt.show()
        except:
            print("can't visualize")

    def __call__(self, x, y=None):
        """
        Perform data preprocessing and return preprocessed data as tuple.

        :param x: Dataset to be preprocessed.
        :type x: `np.ndarray`
        :param y: Labels to be preprocessed.
        :type y: `np.ndarray`
        :return: Preprocessed data
        """
        x = x.astype(np.float32)
        x /= 255.0
        x = self.autoencoder.predict(x)
        x *= 255
        # x = np.around(x)
        return x

    def train(self, x, y=None, callbacks=[], train_epochs_rate=1, start_epoch=0):
        if self.param.get_conf('num_gpu') > 1:
            parallel_model = multi_gpu_model(self.autoencoder, gpus=self.param.get_conf('num_gpu'))
            parallel_model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.001))
        else:
            self.autoencoder.compile(
                loss='binary_crossentropy',
                optimizer=Adam(lr=0.001)
            )


        from keras.callbacks import LearningRateScheduler
        lr_scheduler = LearningRateScheduler(lr_schedule_au)
        callbacks.append(lr_scheduler)

        batch_size = self.param.get_conf()['batch_size']

        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            zca_epsilon=1e-06,  # epsilon for ZCA whitening
            rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
            # randomly shift images horizontally (fraction of total width)
            width_shift_range=5./32,
            # randomly shift images vertically (fraction of total height)
            height_shift_range=5./32,
            shear_range=0.,  # set range for random shear
            zoom_range=0.2,  # set range for random zoom
            channel_shift_range=0.,  # set range for random channel shifts
            # set mode for filling points outside the input boundaries
            fill_mode='nearest',
            cval=0.,  # value used for fill_mode = "constant"
            horizontal_flip=True,  # randomly flip images
            # validation_split=0.0ss
            preprocessing_function=preprocess_autoencoder
        )
        datagen.fit(x)
        def get_fit_encoder(x):
            flow = datagen.flow(x)
            while True:
                batch = next(flow)
                yield (batch, batch)

        gen = get_fit_encoder(x)
        if self.param.get_conf('num_gpu') > 1:
            parallel_model.fit_generator(
                gen,
                epochs=self.param.get_conf('train_epoch_autoencoder') * train_epochs_rate,
                steps_per_epoch=np.ceil(len(x) / batch_size),
                shuffle=True,
                callbacks=callbacks,
                workers=8,
                initial_epoch=start_epoch)
        else:
            self.autoencoder.fit_generator(
                gen,
                epochs=self.param.get_conf('train_epoch_autoencoder') * train_epochs_rate,
                steps_per_epoch=np.ceil(len(x) / batch_size),
                shuffle=True,
                callbacks=callbacks,
                workers=1,
                initial_epoch=start_epoch)

    def get_autoencoder(self):
        return self.autoencoder

    def get_input_tensor(self):
        return self.autoencoder.get_input_at(0)
    
    def get_decoded(self):
        return self.autoencoder.get_layer("decoded").output

    # @staticmethod
    def scheduler(self, epoch):
        lr = K.get_value(self.autoencoder.optimizer.lr)
        for step in self.lr_decay:
            if epoch >= step[0]:
                lr = step[1]
        K.set_value(self.autoencoder.optimizer.lr, lr)
        return K.get_value(self.autoencoder.optimizer.lr)

    def get_lr_metric(self,optimizer):  # printing the value of the learning rate
        def lr(y_true, y_pred):
            return optimizer.lr
        return lr



class AutoEncoderLoss(Layer):
    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(AutoEncoderLoss, self).__init__(**kwargs)

    def call(self, inputs):
        x = Flatten()(inputs[0])
        predict_x = Flatten()(inputs[1])
        noise_x = Flatten()(inputs[2])
        # mse_loss = K.mean(K.square(x - predict_x), axis=-1)
        bin_loss = binary_crossentropy(x, predict_x)
        dis_loss = dot([noise_x - x, noise_x - predict_x], axes=-1, normalize=True)
        loss = K.mean(bin_loss - dis_loss)
        self.add_loss(loss, inputs=inputs)
        # We won't actually use the output.
        return inputs[1]


class AutoEncoderDiffLoss(Layer):
    def __init__(self, alpha=1e-6, **kwargs):
        self.is_placeholder = True
        self.alpha = alpha
        super(AutoEncoderDiffLoss, self).__init__(**kwargs)

    def diffLoss(self, x_1, x_2):
        l2 = 1e-5
        return -1 * mean_squared_error(x_1,x_2) + regularizers.l2(l2)(x_1)+regularizers.l2(l2)(x_2)
        # return -1 * mean_squared_error(x_1,x_2)

    def call(self, inputs):
        x = inputs[0]
        x_decoded = inputs[1]
        loss_1 = K.mean(binary_crossentropy(x, x_decoded))
        loss_2 = 0
        for i in range(2, len(inputs)-1, 2):
            loss_2 += K.mean(self.diffLoss(inputs[i], inputs[i + 1]))
        # l2 = 1
        # loss_2 += K.mean(regularizers.l2(l2)(inputs[-1]))
        loss = loss_1 + self.alpha * loss_2
        self.add_loss(loss, inputs=inputs)
        # We won't actually use the output.
        # return K.expand_dims(loss)
        return loss


class NoiseLayer(Layer):
    def __init__(self, stddevs, mus=0, alpha=1, **kwargs):
        if type(stddevs) == dict:
            self.stddevs = np.array(stddevs['value'])
            self.mus = np.array(mus['value'])
        elif type(stddevs) is list:
            self.stddevs = np.array(stddevs)
            self.mus = np.array(mus)
        else:
            self.stddevs = stddevs
            self.mus = mus
        # self.noises = []
        self.alpha = alpha
        super(NoiseLayer, self).__init__(**kwargs)
    # def build(self, input_shape):
    #     for i in range(input_shape[3]):
    #         self.noises.append(tf.random_normal(shape= np.array([input_shape[1], input_shape[2]]),
    #                mean=self.mus[i], stddev=self.stddevs[i] * self.alpha))
    #     self.noises = tf.stack(self.noises, axis=2)
    #     self.add_weight()
    #     super(NoiseLayer, self).build(input_shape)
    def call(self, x):
        noise = K.random_normal(shape=K.shape(x), mean=self.mus, stddev=self.stddevs * self.alpha)
        return x + noise

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config["stddevs"] = self.stddevs
        config["alpha"] = self.alpha
        config["mus"] = self.mus
        return config

# class AECallback(Callback):
#     def __init__(self, data, data_adversarial, model, model_type='mnist'):
#         super(AECallback, self).__init__()

#         self._model = model
#         self.model_type = model_type
#         self.AE = AE
#         self.data = data
#         self.data_adversarial = data_adversarial
#         # self.osvm = OneClassSVM(nu=0.05, gamma='auto', kernel="rbf")

#     def validate(self):
#         self._model.predict(self.data_adversarial, 'adversarial', self.AE)
#         self._model.predict(self.data, 'clean without defence')
#         self._model.predict(self.data, 'clean', self.AE)

#     def on_epoch_end(self, epoch, logs=None):
#         if self.model_type == 'mnist':
#             print_epoch = 5
#         elif self.model_type == 'cifar':
#             print_epoch = 5
#         if epoch % print_epoch == 0:
#             self.validate()


def GaussiaBlur(input_tensor, kernel_size=3,in_channels=3, **kwargs):
    
    def gauss2D(shape=(3,3),sigma=0.5):
        m,n = [(ss-1.)/2. for ss in shape]
        y,x = np.ogrid[-m:m+1,-n:n+1]
        h = np.exp( -(x*x + y*y) / (2.*sigma*sigma))
        h[ h < np.finfo(h.dtype).eps*h.max()] = 0
        sumh = h.sum()
        if sumh != 0:
            h /= sumh
        return h
    kernel_weights = gauss2D(shape=(kernel_size, kernel_size) )
    
    kernel_weights = np.expand_dims(kernel_weights, axis=-1)
    kernel_weights = np.repeat(kernel_weights, in_channels, axis=-1)
    kernel_weights = np.expand_dims(kernel_weights, axis=-1)
    # print(kernel_weights, kernel_weights.shape)
    g_layer = DepthwiseConv2D(kernel_size, use_bias=False, padding='same', **kwargs)
    output = g_layer(input_tensor)
    g_layer.set_weights([kernel_weights])
    g_layer.trainable = False
    return output

