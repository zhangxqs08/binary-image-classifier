__author__ = 'Xiang'

import numpy as np
from skimage import exposure
import cv2
import glob
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.advanced_activations import ELU
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras import regularizers


def rotate_image(img, angle):
    rows, cols, ch = img.shape
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    return cv2.warpAffine(img, M, (cols, rows))


def load_blur_img(path, img_size, rotate):
    img = cv2.imread(path)
    if type(img) is np.ndarray:
        if rotate:
            angle = np.random.randint(0, 360)
            img = rotate_image(img, angle)
            flip = np.random.randint(0, 2)
            if flip == 1:
                np.fliplr(img)
        img = cv2.blur(img, (5, 5))
        img = cv2.resize(img, img_size)
    return img


def load_img_class(class_path, class_label, class_size, img_size):
    x = []
    y = []
    invalid_imgs = 0
    for path in class_path:
        img = load_blur_img(path, img_size, rotate=True)
        if type(img) is np.ndarray:
            x.append(img)
            y.append(class_label)
        else:
            invalid_imgs += 1

    while len(x) < class_size:
        rand_idx = np.random.randint(0, len(class_path))
        img = load_blur_img(class_path[rand_idx], img_size, rotate=True)
        if type(img) is np.ndarray:
            x.append(img)
            y.append(class_label)
        else:
            invalid_imgs += 1

    print 'invalid_img =', invalid_imgs
    return x, y


def load_data(img_size, class_size):
    hotdogs = glob.glob('./face/*.jpg')
    nothotdogs = glob.glob('./not-face/*.jpg')
    img_size_tuple = (img_size, img_size)
    x_hotdog, y_hotdog = load_img_class(hotdogs, 0, class_size, img_size_tuple)
    x_nothotdog, y_nothotdog = load_img_class(nothotdogs, 1, class_size, img_size_tuple)
    print 'there are', len(x_hotdog), 'face images'
    print 'there are', len(x_nothotdog), 'not face images'
    X = np.array(x_hotdog + x_nothotdog)
    y = np.array(y_hotdog + y_nothotdog)
    return X, y


def gen_grayscale(imgs):
    imgs = 0.2989 * imgs[:, :, :, 0] + 0.587 * imgs[:, :, :, 1] + 0.114 * imgs[:, :, :, 2]
    return imgs


def normalize(imgs):
    imgs = (imgs / 255.0).astype(np.float32)
    for i in range(imgs.shape[0]):
        imgs[i] = exposure.equalize_hist(imgs[i])
        imgs = imgs.reshape(imgs.shape + (1,))
        return imgs


def preprocess_data(imgs):
    gray_imgs = gen_grayscale(imgs)
    return normalize(gray_imgs)


# model building
def keras_model(input_shape):
    model = Sequential()
    model.add(Conv2D(16, (8, 8), padding='valid', input_shape=input_shape))
    model.add(ELU())
    model.add(Conv2D(32, (5, 5), padding='same'))
    model.add(ELU())
    model.add(Conv2D(64, (5, 5), padding='same'))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(ELU())
    model.add(Dense(512, kernel_regularizer=regularizers.l2(1e-05)))
    model.add(Dropout(0.5))
    model.add(ELU())
    model.add(Dense(2))
    model.add(Activation('softmax'))
    return model


def simple_model(input_shape):
    model = Sequential()
    model.add(Conv2D(32, (5, 5), padding='valid', input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (5, 5), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Activation('relu'))
    model.add(Dense(512, kernel_regularizer=regularizers.l2(1e-05)))
    model.add(Dropout(0.3))
    model.add(Activation('relu'))
    model.add(Dense(2))
    model.add(Activation('sigmoid'))
    return model





