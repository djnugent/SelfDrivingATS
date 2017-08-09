import cv2
from keras.models import Sequential, Model
from keras.layers import Lambda, Cropping2D, BatchNormalization, ELU, Merge
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers.convolutional import Convolution2D
from keras.regularizers import l2
from keras import backend as K
import numpy as np


def extract_minimap(img):
    roi = ((469,91),(550,132))
    #crop minimap
    #img = img[91:132,469:550]
    img = img[roi[0][1]:roi[1][1],roi[0][0]:roi[1][0]]
    #YUV colorspace
    img = cv2.cvtColor(img,cv2.COLOR_BGR2YUV)
    # Threshold green navigation arrows
    lower = np.uint8([145,48,0])
    upper = np.uint8([155,58,10])
    img = cv2.inRange(img, lower, upper)
    #resize
    img = cv2.resize(img,None,fx=0.7,fy=0.7)
    # Mean zero
    return np.reshape(img/127.5 - 1.,(29,57,1)), roi


# Prep data before it enters the network
def extract_camera(img):
    roi = ((92,335),(933,651))
    #crop
    #img = img[335:651,92:933]
    img = img[roi[0][1]:roi[1][1],roi[0][0]:roi[1][0]]
    # resize
    img = cv2.resize(img,None,fx=0.35,fy=0.35)
    # YUV colorspace
    img = cv2.cvtColor(img,cv2.COLOR_BGR2YUV)
    # Normalization
    bal = img.astype(np.float64)
    for i in range(0,3):
        #get min, max and range of image
        min_v = np.percentile(bal[:,:,i],2)
        max_v = np.percentile(bal[:,:,i],98)
        #clip extremes
        bal[:,:,i].clip(min_v,max_v, bal[:,:,i])

        #scale image so that brightest pixel is 255 and darkest is 0
        range_v = float(max_v - min_v)
        if range_v > 1:
            bal[:,:,i] = bal[:,:,i] - min_v
            bal[:,:,i] =  bal[:,:,i]*255.0/(range_v)
        else:
            bal[:,:,i] = 0
    # blur
    img = cv2.GaussianBlur(bal,(5,5),0)
    # Mean zero
    img = img/127.5 -1.
    return img, roi

'''
# View extracted data
import imageio
import glob
imgs = glob.glob("E:/dataset/*.png")
for img in imgs:
    img = imageio.imread(img)

    cam = extract_camera(img)
    minimap = extract_minimap(img)
    print(img.shape,cam.shape,minimap.shape)

    cam = cv2.resize(cam,None,fx=3,fy=3,interpolation=cv2.INTER_NEAREST)
    minimap = cv2.resize(minimap,None,fx=3,fy=3,interpolation=cv2.INTER_NEAREST)
    orig = cv2.resize(img,None,fx=0.5,fy=0.5,interpolation=cv2.INTER_NEAREST)

    cv2.imshow("original",orig)
    cv2.imshow("Camera",cam)
    cv2.imshow("minimap",minimap)
    cv2.waitKey(0)
'''

# Custom lost function - Amplifies errors that occur near zero
def mean_precision_error(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true)/(K.abs(y_true) + 0.1), axis=-1)


##### MODEL ####
# Takes in a single image
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

# Takes in an image from the camera and minimap
def v2():
    camera = Sequential()
    # Convolution layers with ELU activation and batch normalization
    camera.add(Convolution2D(24, 5,5, activation='elu', subsample=(2, 2), border_mode='valid',input_shape=(111,294,3)))
    camera.add(BatchNormalization())
    camera.add(Convolution2D(36, 5,5, activation='elu', subsample=(2, 2), border_mode='valid'))
    camera.add(BatchNormalization())
    camera.add(Convolution2D(48, 5,5, activation='elu', subsample=(2, 2), border_mode='valid'))
    camera.add(BatchNormalization())
    camera.add(Convolution2D(64, 3,3, activation='elu', border_mode='valid'))
    camera.add(BatchNormalization())
    camera.add(Flatten())

    minimap = Sequential()
    # Convolution layers with ELU activation and batch normalization
    minimap.add(Convolution2D(32, 3,3, activation='elu', subsample=(2, 2), border_mode='valid',input_shape=(29,57,1)))
    minimap.add(BatchNormalization())
    minimap.add(Convolution2D(64, 3,3, activation='elu', subsample=(2, 2), border_mode='valid'))
    minimap.add(BatchNormalization())
    minimap.add(Flatten())

    model = Sequential()
    # Fully Connected layers with ELU activation and dropout layers
    model.add(Merge([camera, minimap], mode = 'concat'))
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
