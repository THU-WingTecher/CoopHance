# -*- coding:utf-8 -*-

from keras.applications.vgg16 import VGG16

from models.keras_model import KerasClassifier
from utils import *


class VGG16Model():
    def __init__(self, param):
        self.param = param
        with open(self.param.get_conf()['index_path'], 'r') as f:
            self.deocde = json.load(f)
        self.input_shape = param.get_conf('train_image_size')
        input_tensor = Input(( self.input_shape, self.input_shape, 3))
        vgg16 = VGG16(weights='imagenet', input_tensor=input_tensor, include_top=True)
        vgg16.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.classifier = KerasClassifier(model=vgg16, param=self.param,
                                          preprocessing=preprocess_input_vgg)
        # print('self.classifier = ', self.classifier)

    def predict_acc(self, x, y, type_str, defence=None):
        # Evaluate the classifier on the test set
        test_preds = np.argmax(self.classifier.predict(x, defence=defence), axis=1)
        self.test_acc = np.sum(test_preds == y) / y.shape[0]
        print("\n%s accuracy: %.2f%%" % (type_str, self.test_acc * 100))
        return test_preds
        # when result_dict is not empty, start record experiment results

    # to validate backdoor insert effectiveness
    # check whether the backdoor data with poison label is predicted by the model with poison label
    def predict(self, data, type_str='clean', defence=None):
        # Evaluate the classifier on the train set
        K.set_learning_phase(0)
        # Evaluate the classifier on the train set
        train_pred = self.predict(data.x_train, data.y_train, type_str + ' train', defence=defence)

        test_pred = self.predict(data.x_test, data.y_test, type_str + ' test', defence=defence)

        '''
        # visualize predict
        for i in range(3):
            print(np.where(np.array(data.is_poison_test) == 1)[0][i])
            data.visiualize_img_by_idx(np.where(np.array(data.is_poison_test) == 1)[0][i], self.poison_preds[i], False)
        '''
        return train_pred, test_pred

    def predict_instance(self, img):
        # img.shape = (224,224,3)
        # np.expand_dims(img, axis=0).shape = (1,224,224,3)
        # img = preprocess_input_vgg(img)

        pred = self.classifier.predict(img)[0]

        return pred

    def decode_label(self, pred):
        return decode_predictions(pred)[0][0][1]

    def serialize_img(self, img, postfix='gen'):
        self.save_name = '_'.join([self.param.get_conf()['model_prefix'], get_date(), postfix, get_signature()])
        self.save_png = os.path.join(self.param.get_conf()['gen_img_dir'], self.save_name + '.png')
        self.save_pkl = os.path.join(self.param.get_conf()['gen_img_dir'], self.save_name + '.pkl')

        img = deprocess_vgg(img)
        imageio.imsave(uri=self.save_png, im=img)

        # plt.figure()
        # plt.imshow(img)
        # plt.show()

    def decode_index(self, index):
        return self.deocde[str(index)][1]

    def get_input_tensor(self):
        return self.classifier.get_input_tensor()

    def get_output_tensor(self):
        return self.classifier.get_model().get_layer('predictions').output

    def get_output_bef_softmax(self):
        return self.get_classifier().get_output_bef_softmax()

    def get_classifier(self):
        return self.classifier

    def get_input_tensor_origin(self):
        return self.input_tensor



    def dump_model(self):
        serialize_name = '_'.join([self.param.get_conf('model_prefix'), get_date()]) + '.pkl'

        serialize_name = os.path.join(self.param.get_conf()['save_dir'], serialize_name)
        with open(serialize_name, 'wb') as f:
            pickle.dump(self, f)
        print('classifier dump success at', serialize_name)
        return serialize_name

    @staticmethod
    def load_model(model_path):
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print('load classifier success at', model_path)
        return model
