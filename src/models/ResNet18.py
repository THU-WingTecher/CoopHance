# -*- coding: utf-8 -*-
'''ResNet50 model for Keras.
# Reference:
- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
Adapted from code contributed by BigMoyan.
'''
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input
from keras import layers
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import AveragePooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import BatchNormalization
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
import keras.backend as K
from keras.utils import layer_utils
from models.keras_model import KerasClassifier
from keras.optimizers import SGD
from keras.regularizers import l2
from models.CnnModel import CnnModel

from utils import *

def lr_scheduler_sgd(epoch):
    new_lr = 0.1 * (0.1 ** (epoch // 50))
    print('new lr:%.2e' % new_lr)
    return new_lr

def conv2d_bn_relu(x, filters, kernel_size, weight_decay=.0, strides=(1, 1),  name=''):
    layer = conv2d_bn(x, filters, kernel_size, weight_decay, strides, name=name)
    layer = Activation('relu', name='ac_'+name)(layer)
    return layer

def conv2d_bn(x, filters, kernel_size, weight_decay=.0, strides=(1, 1),  name=''):
    layer = Conv2D(filters=filters,
                   kernel_size=kernel_size,
                   strides=strides,
                   padding='same',
                   use_bias=False,
                   kernel_regularizer=l2(weight_decay),
                   name='conv_'+name
                   )(x)
    layer = BatchNormalization(name='bn_'+name)(layer)
    return layer

def ResidualBlock(x, filters, kernel_size, weight_decay, stage, block, downsample=True):
    name_base = 'res' + str(stage) + block + '_branch'
    if downsample:
        # residual_x = conv2d_bn_relu(x, filters, kernel_size=1, strides=2)
        residual_x = conv2d_bn(x, filters, kernel_size=1, strides=2, name=name_base+'1')
        stride = 2
    else:
        residual_x = x
        stride = 1
    residual = conv2d_bn_relu(x,
                              filters=filters,
                              kernel_size=kernel_size,
                              weight_decay=weight_decay,
                              strides=stride,
                              name=name_base+'2a'
                              )
    residual = conv2d_bn(residual,
                         filters=filters,
                         kernel_size=kernel_size,
                         weight_decay=weight_decay,
                         strides=1,
                         name=name_base+'2b'
                         )
    out = layers.add([residual_x, residual], name='add_'+name_base)
    out = Activation('relu', name='ac_'+name_base)(out)
    return out


def ResNet18(classes, input_shape, input_tensor=None, weight_decay=1e-4):
    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    x = img_input
    # x = conv2d_bn_relu(x, filters=64, kernel_size=(7, 7), weight_decay=weight_decay, strides=(2, 2))
    # x = MaxPool2D(pool_size=(3, 3), strides=(2, 2),  padding='same')(x)
    x = conv2d_bn_relu(x, filters=64, kernel_size=(3, 3), weight_decay=weight_decay, strides=(1, 1))

    # # conv 2
    x = ResidualBlock(x, filters=64, kernel_size=(3, 3), weight_decay=weight_decay, 
        downsample=False, stage=2, block='a')
    x = ResidualBlock(x, filters=64, kernel_size=(3, 3), weight_decay=weight_decay, 
        downsample=False, stage=2, block='b')
    # # conv 3
    x = ResidualBlock(x, filters=128, kernel_size=(3, 3), weight_decay=weight_decay,
        downsample=True, stage=3, block='a')
    x = ResidualBlock(x, filters=128, kernel_size=(3, 3), weight_decay=weight_decay, 
        downsample=False, stage=3, block='b')
    # # conv 4
    x = ResidualBlock(x, filters=256, kernel_size=(3, 3), weight_decay=weight_decay, 
        downsample=True, stage=4, block='a')
    x = ResidualBlock(x, filters=256, kernel_size=(3, 3), weight_decay=weight_decay, 
        downsample=False, stage=4, block='b')
    # # conv 5
    x = ResidualBlock(x, filters=512, kernel_size=(3, 3), weight_decay=weight_decay, 
        downsample=True, stage=5, block='a')
    x = ResidualBlock(x, filters=512, kernel_size=(3, 3), weight_decay=weight_decay, 
        downsample=False, stage=5, block='b')
    x = AveragePooling2D(pool_size=(4, 4), padding='valid')(x)
    x = Flatten(name='dense_2')(x)
    x = Dense(classes, activation=None, name='predictions')(x)
    x = Activation('softmax', name='softmax_output')(x)
    # model = Model(img_input, x, name='ResNet18')
    return img_input, x


class ResNet(CnnModel):

    def __init__(self, param, input_tensor=None):
        super(ResNet, self).__init__(param)
        self.init_model(input_tensor = input_tensor)
        self.lr_scheduler_fun = lr_scheduler_sgd
    
    def init_model(self,include_top=False, weights=None,
                input_tensor=None, input_shape=None,
                pooling=None):
        """Instantiates the ResNet50 architecture.
        Optionally loads weights pre-trained
        on ImageNet. Note that when using TensorFlow,
        for best performance you should set
        `image_data_format="channels_last"` in your Keras config
        at ~/.keras/keras.json.
        The model and the weights are compatible with both
        TensorFlow and Theano. The data format
        convention used by the model is the one
        specified in your Keras config file.
        # Arguments
            include_top: whether to include the fully-connected
                layer at the top of the network.
            weights: one of `None` (random initialization)
                or "imagenet" (pre-training on ImageNet).
            input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
                to use as image input for the model.
            input_shape: optional shape tuple, only to be specified
                if `include_top` is False (otherwise the input shape
                has to be `(224, 224, 3)` (with `channels_last` data format)
                or `(3, 224, 244)` (with `channels_first` data format).
                It should have exactly 3 inputs channels,
                and width and height should be no smaller than 197.
                E.g. `(200, 200, 3)` would be one valid value.
            pooling: Optional pooling mode for feature extraction
                when `include_top` is `False`.
                - `None` means that the output of the model will be
                    the 4D tensor output of the
                    last convolutional layer.
                - `avg` means that global average pooling
                    will be applied to the output of the
                    last convolutional layer, and thus
                    the output of the model will be a 2D tensor.
                - `max` means that global max pooling will
                    be applied.
            classes: optional number of classes to classify images
                into, only to be specified if `include_top` is True, and
                if no `weights` argument is specified.
        # Returns
            A Keras model instance.
        # Raises
            ValueError: in case of invalid argument for `weights`,
                or invalid input shape.
        """


        # Determine proper input shape
        # input_shape = _obtain_input_shape(self.input_shape,
        #                                 default_size=224,
        #                                 min_size=32, #for cifar10 compatibility;
        #                                 data_format=K.image_data_format(),
        #                                 require_flatten=include_top) #Look keras 2.0+ version change logs

        img_input, x = ResNet18(self.param.get_conf("classes_num"), input_tensor=input_tensor, input_shape=self.input_shape)
        # Ensure that the model takes into account
        # any potential predecessors of `input_tensor`.
        # if input_tensor is not None:
        #     inputs = get_source_inputs(input_tensor)
        # else:
        #     inputs = img_input
        # Create model.

        model = Model(img_input, x, name='resnet18')
        # model.summary()
        # load weights
        # weights_path = '../model/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
        # model.load_weights(weights_path,strict=False)


        opt = SGD(lr=0.1, momentum=0.9, nesterov=False)
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        self.classifier = KerasClassifier(clip_values=(0, 255), model=model, param=self.param, preprocessing=preprocess_mnist)
        
    # def train(self, data):
    #     # default
    #     # nb_epochs=20
    #     # batch_size=128
    #     from keras.preprocessing.image import ImageDataGenerator

    #     # if self.model_name == 'cnn':
    #     #     datagen = ImageDataGenerator(zoom_range=0.2, horizontal_flip=True)
    #     #     # zoom 0.2
    #     #     datagen = create_resnet_generator(dataset.x_train)
    #     #     callbacks_list = []
    #     #     batch_size = 128
    #     #     num_epochs = 200

     
    #     from keras.callbacks import LearningRateScheduler, ReduceLROnPlateau, EarlyStopping
    #     # zoom 0.2 horizontal_filp always True. change optimizer to sgd, and batch_size to 128. 
    #     datagen = ImageDataGenerator(rotation_range=15,
    #                             width_shift_range=5./32,
    #                             height_shift_range=5./32,
    #                             horizontal_flip = True,
    #                             zoom_range = 0.2,
    #                             preprocessing_function=preprocess_mnist)

    #     datagen.fit(data.x_train, seed=0)


    #     from keras.callbacks import LearningRateScheduler
    #     lr_scheduler = LearningRateScheduler(lr_scheduler_sgd)
    #     callbacks_list = [lr_scheduler]
    #     batch_size = self.param.get_conf('batch_size')
    #     num_epochs = self.param.get_conf('train_epoch')

    #     # filepath="{}/models/original.hdf5".format(self.data_model)
    #     # checkpoint = ModelCheckpoint(filepath, monitor='val_acc', 
    #     #     verbose=1, save_best_only=True, mode='max')
    #     # callbacks_list.append(checkpoint)

        
    #     model_info = self.classifier.get_model().fit_generator(datagen.flow(data.x_train, 
    #         to_categorical(data.y_train), batch_size = batch_size),
    #         epochs = num_epochs,
    #         steps_per_epoch = data.x_train.shape[0] // batch_size,
    #         callbacks = callbacks_list, 
    #         validation_data = (data.x_test, to_categorical(data.y_test)), 
    #         verbose = 2,
    #         workers = 4)

    # def predict_acc(self, x, y, type_str, defence=None):
    #     # Evaluate the classifier on the test set
    #     test_preds = np.argmax(self.classifier.predict(x, defence=defence), axis=1)
    #     self.test_acc = np.sum(test_preds == y) / y.shape[0]
    #     print("\n%s accuracy: %.2f%%" % (type_str, self.test_acc * 100))
    #     return test_preds

    # def predict_acc(self, x, y, type_str, defence=None):
    #     # Evaluate the classifier on the test set
    #     test_preds = np.argmax(self.classifier.predict(x, defence=defence), axis=1)
    #     self.test_acc = np.sum(test_preds == y) / y.shape[0]
    #     print("\n%s accuracy: %.2f%%" % (type_str, self.test_acc * 100))
    #     return test_preds
        # when result_dict is not empty, start record experiment results

    # to validate backdoor insert effectiveness
    # check whether the backdoor data with poison label is predicted by the model with poison label
    # def predict(self, data, type_str='clean', defence=None):
    #     # Evaluate the classifier on the train set
    #     K.set_learning_phase(0)
    #     # Evaluate the classifier on the train set
    #     train_pred = self.predict(data.x_train, data.y_train, type_str + ' train', defence=defence)

    #     test_pred = self.predict(data.x_test, data.y_test, type_str + ' test', defence=defence)

    #     '''
    #     # visualize predict
    #     for i in range(3):
    #         print(np.where(np.array(data.is_poison_test) == 1)[0][i])
    #         data.visiualize_img_by_idx(np.where(np.array(data.is_poison_test) == 1)[0][i], self.poison_preds[i], False)
    #     '''
    #     return train_pred, test_pred

    # def predict_instance(self, x):
    #     return self.classifier.predict(x)[0]

    # def get_input_shape(self):
    #     return self.input_shape

    # def set_input_shape(self, input_shape):
    #     self.input_shape = input_shape

    # def get_classifier(self):
    #     return self.classifier

    # def set_classifier(self, classifier):
    #     self.classifier = classifier

    # def get_input_tensor(self):
    #     return self.classifier.get_input_tensor()

    # def get_output_tensor(self):
    #     return self.classifier.get_output_tensor()

    # def get_output_bef_softmax(self):
    #     return self.classifier.get_output_bef_softmax()


    # @staticmethod 
    # def load_model(model_path):
    #     K.set_learning_phase(0)
    #     with open(model_path, 'rb') as f:
    #         model = pickle.load(f)
    #     print('load classifier success at', model_path)
    #     return model
    
    # def dump_model(self):
    #     serialize_name = '_'.join([self.param.get_conf('model_prefix'), get_date()]) + '.pkl'

    #     serialize_name = os.path.join(self.param.get_conf()['save_dir'], serialize_name)
    #     with open(serialize_name, 'wb') as f:
    #         pickle.dump(self, f)
    #     print('classifier dump success at', serialize_name)
    #     return serialize_name




    # def collect_neurals(self, x, nerual_lists, defence = None):
    #     assert type(nerual_lists) is list
    #     res = [[] for n in nerual_lists]
    #     req_tensors = [self.classifier._model.get_layer(na).output for na in nerual_lists]
    #     batch_size = self.param.get_conf()['batch_size']
    #     fun = K.function([self.classifier._model.input],req_tensors)
    #     len_batch= int(np.ceil(len(x) / float(batch_size)))
    #     for batch_index in range(len_batch):
    #         begin, end = batch_index * batch_size, min((batch_index + 1) * batch_size, len(x))
    #         if type(x[0]) is str:
    #             raw_x = []
    #             # print('path = ', path)
    #             # print('self.data_path = ', self.data_path)
    #             for p in range(begin, end):
    #                 img = cv2.imread(x[p])[:, :, ::-1]
    #                 img = cv2.resize(img, self.train_shape)
    #                 raw_x.append(img)
    #             # import pdb
    #             # pdb.set_trace()

    #             pre_x = preprocess_autoencoder(np.array(raw_x))
    #         else:
    #             raw_x = x[begin:end]
    #             pre_x = preprocess_autoencoder(raw_x)
    #         if defence is not None:
    #             pre_x = defence(pre_x)
    #         neruals = fun([pre_x])
    #         for i in range(len(nerual_lists)):
    #             res[i].append(neruals[i])
    #     for i in range(len(nerual_lists)):
    #         res[i] = np.concatenate(res[i], axis=0)
    #     return res

if __name__ == '__main__':
    model = ResNet50(include_top=True, weights='imagenet')

    img_path = 'elephant.jpg'
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    print('Input image shape:', x.shape)

    preds = model.predict(x)
    print('Predicted:', decode_predictions(preds))