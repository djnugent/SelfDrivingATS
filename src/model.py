import cv2
from keras.models import Sequential, Model
from keras.layers import Lambda, Cropping2D, BatchNormalization, ELU
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers.convolutional import Convolution2D
from keras.regularizers import l2
from keras import backend as K


# Preprocess data before it enters the network
def preprocess(img):
    #crop
    img = img[335:651,92:933]
    #resize
    img = cv2.resize(img,None,fx=0.35,fy=0.35)
    #blur
    img = cv2.GaussianBlur(img,(5,5),0)
    #YUV colorspace
    img = cv2.cvtColor(img,cv2.COLOR_BGR2YUV)
    # Mean zero
    return img/127.5 - 1.


# Custom lost function - Amplifies errors that occur near zero
def mean_precision_error(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true)/(K.abs(y_true) + 0.1), axis=-1)


##### MODEL ####
def v1():
    model = Sequential()
    # Convolution layers with ELU activation and batch normalization
    model.add(Convolution2D(24, 5,5, activation='elu', subsample=(2, 2), border_mode='valid',input_shape=(111,294,3)))
    model.add(BatchNormalization())
    model.add(Convolution2D(36, 5,5, activation='elu', subsample=(2, 2), border_mode='valid'))
    model.add(BatchNormalization())
    model.add(Convolution2D(48, 5,5, activation='elu', subsample=(2, 2), border_mode='valid'))
    model.add(BatchNormalization())
    model.add(Convolution2D(64, 3,3, activation='elu', border_mode='valid'))
    model.add(BatchNormalization())
    # Fully Connected layers with ELU activation and dropout layers
    model.add(Flatten())
    model.add(Dropout(0.50))
    model.add(Dense(100, activation='elu'))
    model.add(Dropout(0.50))
    model.add(Dense(50, activation='elu'))
    model.add(Dropout(0.50))
    model.add(Dense(10, activation='elu'))
    model.add(Dropout(0.50))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    return model
