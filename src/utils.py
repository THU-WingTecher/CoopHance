# -*- coding:utf-8 -*-

from conf import *


def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)


def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(28, 28), grayscale=True)
    input_img_data = image.img_to_array(img)
    input_img_data = input_img_data.reshape(1, 28, 28, 1)

    input_img_data = input_img_data.astype('float32')
    input_img_data /= 255

    # input_img_data = preprocess_input(input_img_data)  # final input shape = (1,224,224,3)
    return input_img_data


def get_signature():
    now = datetime.datetime.now()
    past = datetime.datetime(2015, 6, 6, 0, 0, 0, 0)
    timespan = now - past
    time_sig = int(timespan.total_seconds() * 1000)

    return str(time_sig)


def serialize_img(img, param):
    save_name = '_'.join([
        param.get_conf()['model_prefix'],
        get_date(), 'image',
        get_signature()
    ])
    save_path = os.path.join(param.get_conf()['perturbation_dir'],
                             save_name + '.png')
    save_pkl = os.path.join(param.get_conf()['perturbation_dir'],
                            save_name + '.pkl')

    img = img.flatten().reshape((28, 28))
    print('img.shape = ', img.shape)

    plt.figure()
    plt.imshow(img, cmap='gray')
    plt.show()

    imageio.imwrite(uri=save_path, im=img)

    with open(save_pkl, 'wb') as f:
        pickle.dump(img, f)

    print('save img done')


def deserialize_pert(save_pkl, alpha):
    with open(save_pkl, 'rb') as f:
        perturb = pickle.load(f)

    print('load perturbation done')
    print('self.perturb.shape = ', perturb.shape)

    print('alpha = ', alpha)

    perturb = perturb * alpha
    # cilp the float part, 3.7->4, 3.1->3
    # perturb = (perturb*255).astype(np.int32)
    # perturb = perturb.astype(np.uint8)

    return perturb


def to_categorical(labels, nb_classes=None):
    """
    Convert an array of labels to binary class matrix.

    :param labels: An array of integer labels of shape `(nb_samples,)`
    :type labels: `np.ndarray`
    :param nb_classes: The number of classes (possible labels)
    :type nb_classes: `int`
    :return: A binary matrix representation of `y` in the shape `(nb_samples, nb_classes)`
    :rtype: `np.ndarray`
    """
    labels = np.array(labels, dtype=np.int32)
    if not nb_classes:
        nb_classes = np.max(labels) + 1
    categorical = np.zeros((labels.shape[0], nb_classes), dtype=np.float32)
    categorical[np.arange(labels.shape[0]), np.squeeze(labels)] = 1
    return categorical


def preprocess_autoencoder(x):
    x = x.astype(np.float32)
    return x / 255.0


def preprocess_mnist(x, y=None, nb_classes=10, clip_values=None):
    """Scales `x` to [0, 1] and converts `y` to class categorical confidences.

    :param x: Data instances.
    :type x: `np.ndarray`
    :param y: Labels.
    :type y: `np.ndarray`
    :param nb_classes: Number of classes in dataset.
    :type nb_classes: `int`
    :param clip_values: Original data range allowed value for features, either one respective scalar or one value per
           feature.
    :type clip_values: `tuple(float, float)` or `tuple(np.ndarray, np.ndarray)`
    :return: Rescaled values of `x`, `y`
    :rtype: `tuple`
    """
    # if clip_values is None:
    #     min_, max_ = np.amin(x), np.amax(x)
    # else:
    #     min_, max_ = clip_values
    #
    # normalized_x = (x - min_) / (max_ - min_)
    normalized_x = x.astype(np.float32) / 255.0
    if y is not None:
        categorical_y = to_categorical(y, nb_classes)

        return normalized_x, categorical_y
    return normalized_x


def deprocess_mnist(x, y=None):
    x = x * 255.0
    x = np.clip(x, 0, 255)
    if y is not None:
        y = np.argmax(y, axis=1)
        return x, y
    return x


def preprocess_input_vgg(x):

    if (len(x.shape) == 3):
        x = np.expand_dims(x, axis=0)
    # x = cv2.resize(x, (224, 224))
    x = x.astype(np.float32)
    x = preprocess_input(x)
    return x


def deprocess_vgg(x):
    x = x.reshape((224, 224, 3))
    # Remove zero-center by mean pixel
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    # 'BGR'->'RGB'
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255)
    return x


def load_img(img_path):
    img = cv2.imread(img_path)[:, :, ::-1]
    img = cv2.resize(img, (224, 224))
    img = preprocess_input_vgg(img)
    return img


def dump_model(model, param, prefix='model_prefix'):
    # concat dump name
    serialize_name = '_'.join([param.get_conf()[prefix], get_date()]) + '.pkl'
    print('serialize_name = ', serialize_name)

    # concat dump path
    serialize_path = os.path.join(param.get_conf()['save_dir'], serialize_name)
    with open(serialize_path, 'wb') as f:
        pickle.dump(model, f)

    print('model dump success')

    return serialize_path


def deserialize_model(path):
    with open(path, 'rb') as f:
        model = pickle.load(f)

    print('model load success')

    return model


def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def cross_entropy_error(y,t):
    if y.ndim==1:
        t=t.reshape(t.size, 1)
        y=y.reshape(y.size, 1)
    elif y.ndim > 2:
        t = t.reshape(len(t), -1)
        y = y.reshape(len(t), -1)
    return np.mean(np.nan_to_num(-t*np.log(y)-(1-t)*np.log(1-y)))


class Param:
    def __init__(self, json_file, prefix=None):
        self.conf = None
        self.json_file = json_file
        if prefix:
            self.json_path = os.path.join(prefix, self.json_file)
        else:
            self.json_path = os.path.join(json_dir, self.json_file)

        with open(self.json_path, 'r') as f:
            self.conf = json.load(f)

        for key, val in self.conf.items():
            print(key, ':', val)

    def get_conf(self, key=None):
        if key == None:
            return self.conf
        else:
            return self.conf[key]

    def set_conf(self, key, value):
        self.conf[key] = value

    def dump_conf(self, path=None):
        if path is None:
            with open(self.json_path, 'w') as f:
                json.dump(self.conf, f, sort_keys=False, indent=2)
        else:
            with open(path, 'w') as f:
                json.dump(self.conf, f, sort_keys=False, indent=2)


def lr_schedule_au(epoch, logs=None):
    decay = epoch >= 300 and 2 or epoch >= 180 and 1 or 0
    lr = 1e-1 * 0.1 ** decay
    print('Learning rate: ', lr)
    return lr	
