# - * - coding: utf - 8 -
# *-
from keras.engine import Layer
from keras.layers.merge import dot
from keras.losses import binary_crossentropy
from tqdm.gui import trange
from utils import *
from defences.autoencoder_abstract import NoiseLayer

class OsvmCallback(Callback):
    def __init__(self, data, _model, act_layer, model_type='mnist'):
        super(Callback, self).__init__()
        self.x_clean_train, _, self.x_clean_test, _ = data.get_specific_label_clean_data(
            6)
        self.x_poison_train, _, self.x_poison_test, _ = data.get_specific_label_poison_data(
            6)
        # self.x_clean_train = self.x_clean_train[:25000]
        self._model = _model
        self.act_layer = act_layer
        self.model_type = model_type

        # self.osvm = OneClassSVM(nu=0.05, gamma='auto', kernel="rbf")

    def get_act(self, x):
        # print(self.model,type(self.x_clean_train))
        [self.decoded_imgs, self.predictions] = self._model.predict(x)
        # self.decoded_imgs =  self.model.predict(self.x_clean_train[:2500])
        self.act = K.function(
            [self._model.get_input_at(0)], [
                self._model.get_layer(
                    self.act_layer).output
            ])([self.x])[0]
        self.act_flatten = np.reshape(self.act,
                                      (-1, np.prod(self.act.shape[1:])))

    def pred(self, x, y, data_name, vis=False):
        self.get_act(x)
        preds = self.osvm.predict(self.act_flatten)
        a = np.sum(preds == y)
        b = np.sum(preds != y)

        return a, b, a / len(x)

    def on_epoch_end(self, epoch, logs=None):
        if self.model_type == 'mnist':
            print_epoch = 3
        elif self.model_type == 'cifar':
            print_epoch = 3
        if epoch % print_epoch == 0:
            self.osvm_f1()


class EncoderClassifier_abstract():
    def __init__(self, param):
        self.autoencoder = None
        self.encoded_imgs = None
        self.param = param
        self.mdoel = None

    def __getstate__(self):
        """
        Use to ensure `KerasClassifier` can be pickled.

        :return: State dictionary with instance parameters.
        :rtype: `dict`
        """

        state = self.__dict__.copy()

        # Remove the unpicklable entries

        del state['autoencoder']
        del state['model']
        del state['encoded_imgs']
        del state['decoded_imgs']

        # model_name = str(time.time()) + '.h5'
        model_name = '_'.join(['encoder_classifier', get_date()]) + '.h5'
        state['model_name'] = model_name
        self.save(model_name, path=self.param.get_conf()['save_dir'])
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

        full_path = os.path.join(self.param.get_conf()['save_dir'],
                                 state['model_name'])
        self.model = load_model(str(full_path), custom_objects={'NoiseLayer': NoiseLayer})
        self.autoencoder = Model(inputs=self.model.get_input_at(0),
                                 output=self.model.get_layer("decoded").output)
        # self._initialize_params()
        # self._model = self.model
        # if self.param.get_conf()['num_gpu'] > 1:
        #     self.model = multi_gpu_model(self.model, gpus=self.param.get_conf()['num_gpu'])
        # self._initialize_params()

    def dump_model(self):
        name = '_'.join(
            ['encoder',
             self.param.get_conf()['model_prefix'],
             get_date()]) + '.pkl'
        path = os.path.join(self.param.get_conf()['save_dir'], name)
        if not os.path.exists(self.param.get_conf()['save_dir']):
            os.makedirs(self.param.get_conf()['save_dir'])
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        print("encoder classifier dump success at " + path)
        return name
    
        
     

    def dump_autoencoder(self):
        name = '_'.join(['autoencoder', self.param.get_conf()['model_prefix'], get_date()]) + '.pkl'
        path = os.path.join(self.param.get_conf()['save_dir'], name)
        if not os.path.exists(self.param.get_conf()['save_dir']):
            os.makedirs(self.param.get_conf()['save_dir'])
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        print("autoencoder dump success at " + path)
        return name

    def get_model(self):
        return self.model

    def save(self, filename, path=None):
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

        self.model.save(str(full_path))
        logger.info('Model saved in path: %s.', full_path)

    
    @staticmethod
    def load_model(path):
        K.set_learning_phase(0)
        with open(path, 'rb') as f:
            model = pickle.load(f)
            print('load encoder classifier success at ' + path)
        return model

    def evaluate(self, data, type_str='normal', vis=False):
        K.set_learning_phase(0)
        # Evaluate the classifier on the train set
        self.predict(data.x_train, vis=False)
        acc = np.mean(data.y_train == self.predicts)
        print("Accurancy on", type_str, "train set is %.4f" % acc)

        self.predict(data.x_test, vis=vis)
        acc = np.mean(data.y_test == self.predicts)
        print("Accurancy on", type_str, "test set is %.4f" % acc)

    def predict(self, x,y=None, type_str='', vis=False):

        decoded_imgs = np.zeros_like(x, dtype=np.float32)
        self.predicts = np.zeros((len(x), self.param.get_conf('classes_num')), dtype=np.float32)
        batch_size = self.param.get_conf()['batch_size']
        vis_img = None
        rr = int(np.ceil(len(x) / float(batch_size)))
        for batch_index in range(rr):
            begin, end = batch_index * batch_size, min(
                (batch_index + 1) * batch_size, len(x))
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
            # decoded = self.autoencoder.predict(pre_x) * 255
            [decodeds, preds] = self.model.predict(pre_x)
            decodeds *= 255.0
            decoded_imgs[begin:end] = decodeds
            self.predicts[begin:end] = preds
            if vis_img is None:
                vis_img = raw_x
        decoded_imgs = np.squeeze(np.array(decoded_imgs))
        if vis:
            self.visualize_autoencoder(vis_img, decoded_imgs, type_str)
        self.predicts = self.predicts.argmax(axis=1)
        if y:
            print("Accurancy on", type_str, "set is %.4f" % np.mean(self.predicts == y))
        return self.predicts

    def visualize_autoencoder(self, x_test, decoded_imgs, type_str='normal'):
        n = 10
        x_test = x_test[:n + 1].astype(np.uint8)
        x_decoded = decoded_imgs[:n + 1]
        # x_decoded = deprocess_autoencoder(x_decoded)
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
        plt.savefig(
            os.path.join(self.param.get_conf('log_dir'), 'decoded',
                         'decoded_' + get_date() + '.jpg'))

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

                plt.imshow(self.encoded_imgs[i][:, :, :32].reshape(4,
                                                                   4 * 32).T)
                # plt.gray()
                # ax.get_xaxis().set_visible(False)
                # ax.get_yaxis().set_visible(False)
            plt.show()
        except:
            print("can't visualize")

    # def __call__(self, x, y=None):
    #     """
    #     Perform data preprocessing and return preprocessed data as tuple.

    #     :param x: Dataset to be preprocessed.
    #     :type x: `np.ndarray`
    #     :param y: Labels to be preprocessed.
    #     :type y: `np.ndarray`
    #     :return: Preprocessed data
    #     """
    #     x = x.astype(np.float32)
    #     x /= 255.0
    #     x = self.autoencoder.predict(x)
    #     x *= 255
    #     # x = np.around(x)
    #     return x


    def get_input_tensor(self):
        return self.model.get_input_at(0)

    def get_output_tensor(self):
        return self.model.get_output_at(0)[1]

    def get_output_bef_softmax(self):
        return self.model.get_layer(
            "predictions").output

    def get_encoder_classifier(self):
        return self.model

    def get_layer_output(self, layer_name):
            return self.model.get_layer(layer_name).output

    def train(self, x, y):
        pass


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
        dis_loss = dot([noise_x - x, noise_x - predict_x],
                       axes=-1,
                       normalize=True)
        loss = K.mean(bin_loss - dis_loss)
        self.add_loss(loss, inputs=inputs)
        # We won't actually use the output.
        return inputs[1]


