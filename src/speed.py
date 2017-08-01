from capnctrl import cap
import cv2
import time
import glob
import numpy as np
import os
from sklearn import svm
from sklearn.externals import joblib


class speed:
    def __init__(self):
        self.svm = None

    def load(self):
        self.svm = joblib.load("ocr.pkl")

    def train_svm(self,regex="*-*.png"):

        files = glob.glob(regex)
        x_train = []
        y_train = []
        for f in files:
            img = cv2.imread(f)
            img = img[:,:,0]
            x_train.append(self.norm(img).flatten())
            y_train.append(int(f[0]))

        clf = svm.SVC(decision_function_shape='ovo')
        clf.fit(x_train,y_train)
        print(clf.score(x_train,y_train))

        joblib.dump(clf, 'ocr.pkl')

    def label_dataset(self,regex="test*.png"):
        #os.mkdir("dataset")
        files = glob.glob(regex)
        cnt = 0
        for f in files:
            img = cv2.imread(f)
            digits = self.extract_digits(img)
            for dig in digits:
                cv2.imshow("img",cv2.resize(dig,(16,28),interpolation=cv2.INTER_NEAREST ))
                cv2.waitKey(1)
                user = input("What is this digit: ")
                cv2.imwrite("{}-{}.png".format(user,cnt),dig)
                cnt +=1


    def extract_digits(self,img):
        # Convert tp YCrCb
        img = cv2.cvtColor(img,cv2.COLOR_BGR2YCrCb)
        # Extract Y channel
        y = img[:,:,0]
        # threshold
        #idx = y > 170
        #binary = np.zeros_like(y)
        #binary[idx] = 255
        binary = y
        digits = []
        # Determine if 1 or 2 digits by last column of pixels
        # the last column is brighter if it is a single digit
        col_sum = np.sum(binary[:,-1])

        # Single digit
        if col_sum > 1000:
            digits.append(binary[:,3:7])
        # two digits
        else:
            digits.append(binary[:,1:5])
            digits.append(binary[:,6:10])

        return digits

    def norm(self,img):
        mi = img.min()
        ma = img.max()
        if mi == ma:
            return np.zeros_like(img)
        return (img - mi) * 1.0/(ma-mi)

    def predict(self,img):
        digits = self.extract_digits(img)
        if len(digits) == 2:
            d0 = self.norm(digits[0]).flatten()
            d1 = self.norm(digits[1]).flatten()
            pred = self.svm.predict([d0,d1])
            spd = pred[0] * 10 + pred[1]
        else:
            d0 = self.norm(digits[0]).flatten()
            pred = self.svm.predict([d0])
            spd = pred[0]
        return spd

if __name__=="__main__":

    spd = speed()
    #spd.label_dataset()
    spd.train_svm()
