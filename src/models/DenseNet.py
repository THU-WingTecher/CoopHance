from models.CnnModel import CnnModel, lr_scheduler_adv
from models.keras_model import KerasClassifier
from utils import *
import keras
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Flatten, Input, AveragePooling2D, merge, Activation, SpatialDropout2D
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, SeparableConv2D
from keras.layers import Concatenate
from keras.optimizers import SGD
from keras import regularizers
import tensorflow as tf
import keras.backend as K

def lr_scheduler_sgd(epoch):
    new_lr = 0.1 * (0.1 ** (epoch // 50))
    print('new lr:%.2e' % new_lr)
    return new_lr 

def add_denseblock(input, stage, num_filter = 12, dropout_rate = 0.2, compression=0.5, l=12):
    temp = input
   
    for _ in range(l):
        base_name = 'dense'+str(stage)+'_layer'+str(_)
        BatchNorm = BatchNormalization(name = base_name+'_bn')(temp)
        relu = Activation('relu', name=base_name+'_ac')(BatchNorm)
        Conv2D_3_3 = Conv2D(int(num_filter*compression), (3,3), use_bias=False
                 ,padding='same', name=base_name+'_conv')(relu)
        #if dropout_rate>0:
         # Conv2D_3_3 = Dropout2D(dropout_rate)(Conv2D_3_3)
        concat = Concatenate(axis=-1)([temp,Conv2D_3_3])
        
        temp = concat
        
    return temp

def add_transition(input, stage, num_filter = 12, dropout_rate = 0.2, compression=0.5):
    base_name = 'dense'+str(stage)+'_trans'
    BatchNorm = BatchNormalization(name = base_name+'_bn')(input)
    relu = Activation('relu', name = base_name+'_ac')(BatchNorm)
    Conv2D_BottleNeck = Conv2D(int(num_filter*compression), (1,1), use_bias=False,
        kernel_regularizer = regularizers.l1() ,padding='same', name = base_name+'_conv')(relu)
    #if dropout_rate>0:
      #Conv2D_BottleNeck = Dropout2D(dropout_rate)(Conv2D_BottleNeck)
    avg = AveragePooling2D(pool_size=(2,2), name = base_name+'_pool')(Conv2D_BottleNeck)
    
    return avg

def output_layer(input, num_classes):
    BatchNorm = BatchNormalization()(input)
    relu = Activation('relu')(BatchNorm)
    AvgPooling = AveragePooling2D(pool_size=(2,2), name='dense_2')(relu)
    temp = Conv2D(num_classes, kernel_size = (2,2))(AvgPooling)
    flat = Flatten(name = 'predictions')(temp)
    output = Activation('softmax', name='softmax_output')(flat)

    
    return output

def densenet(num_classes, input_shape, input_tensor=None, num_filter = 36, dropout_rate=0.2, compression=0.5, l=12):
    if input_tensor is None:
            img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    First_Conv2D = Conv2D(num_filter, (3,3), use_bias=False ,padding='same', name='conv1')(img_input)

    First_Block = add_denseblock(First_Conv2D, num_filter=num_filter, 
        dropout_rate=dropout_rate, compression=compression, l=l, stage=1)
    First_Transition = add_transition(First_Block, num_filter=num_filter,
        dropout_rate=dropout_rate, compression=compression, stage=1)

    Second_Block = add_denseblock(First_Transition, num_filter=num_filter,
        dropout_rate=dropout_rate, compression=compression, l=l, stage=2)
    Second_Transition = add_transition(Second_Block, num_filter=num_filter,
        dropout_rate=dropout_rate, compression=compression, stage=2)

    Third_Block = add_denseblock(Second_Transition, num_filter=num_filter,
        dropout_rate=dropout_rate, compression=compression, l=l, stage=3)
    Third_Transition = add_transition(Third_Block, num_filter=num_filter,
        dropout_rate=dropout_rate, compression=compression, stage=3)

    Last_Block = add_denseblock(Third_Transition,  num_filter=num_filter,
        dropout_rate=dropout_rate, compression=compression, l=l, stage=4)
    output = output_layer(Last_Block, num_classes)
    return img_input, output

class DenseNet(CnnModel):
    def __init__(self, param, input_tensor=None, dropout_rate=0.2, l = 12, num_filter = 36, compression = 0.5): #added 24 more filters
        super(DenseNet, self).__init__(param)
        self.dropout_rate=dropout_rate
        self.compression = compression
        self.num_filter= num_filter
        self.l=l
        self.lr_scheduler_fun = lr_scheduler_sgd
        self.init_model(input_tensor = input_tensor, input_shape=self.input_shape)

    def init_model(self, input_shape, input_tensor=None):
        
        # model.summary()
        # load weights
        # weights_path = '../model/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
        # model.load_weights(weights_path,strict=False)
        img_input, output = densenet(num_classes=self.param.get_conf('classes_num'), input_shape=input_shape, 
                compression=self.compression, num_filter=self.num_filter, dropout_rate=self.dropout_rate, l=self.l)
        model = Model(img_input, output, name='DenseNet')
        self.lr_scheduler_fun = lr_scheduler_adv
        opt = SGD(lr=0.1, momentum=0.9, nesterov=False)
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        self.classifier = KerasClassifier(clip_values=(0, 255), model=model, param=self.param, preprocessing=preprocess_mnist)
        return         

