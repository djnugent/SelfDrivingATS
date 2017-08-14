import cv2
import numpy as np
import imageio
import glob

files = glob.glob("D:/dataset/*.png")

def transform(img):
    h,w,c = img.shape

    lane_width = 130 #85
    apex_width = 80 #230
    horizon0 = 415 #485
    horizon1 = 0 #610
    src = np.float32([[0,h],
                    [w,h],
                    [int(w/2 - apex_width/2),horizon0],
                    [int(w/2 + apex_width/2),horizon0]])
    dst = np.float32([[int(w/2 - lane_width/2), h],
                      [int(w/2 + lane_width/2), h],
                      [int(w/2 - lane_width/2), horizon1],
                      [int(w/2 + lane_width/2), horizon1]])
    # Calculate the transform
    M = cv2.getPerspectiveTransform(src, dst)
    inv_M = cv2.getPerspectiveTransform(dst, src)
    result = cv2.warpPerspective(img, M, (w,h))
    return result

def rgbnorm(img):
    h,w,c = img.shape
    norm = np.zeros_like(img)
    for y in range(h):
        for x in range(w):
            np.sum(img,axis=2)

for f in files:
    img = imageio.imread(f)
    mul = 255.0 / np.sum(img.astype(np.float32),axis=2)
    img = img.astype(np.float32)
    img[:,:,0] = np.multiply(img[:,:,0],mul)
    img[:,:,1] = np.multiply(img[:,:,1],mul)
    img[:,:,2] = np.multiply(img[:,:,2],mul)
    print(img.shape)
    '''
    img = cv2.cvtColor(img,cv2.COLOR_BGR2YCrCb)

    # blur
    img = cv2.GaussianBlur(img,(5,5),0)

    # Normalization
    roi = ((92,335),(933,651))
    #crop
    bal = img.astype(np.float64)
    for i in range(0,3):
        #get min, max and range of image
        min_v = np.percentile(bal[roi[0][1]:roi[1][1],roi[0][0]:roi[1][0],i],2)
        max_v = np.percentile(bal[roi[0][1]:roi[1][1],roi[0][0]:roi[1][0],i],98)
        #clip extremes
        bal[:,:,i].clip(min_v,max_v, bal[:,:,i])

        #scale image so that brightest pixel is 255 and darkest is 0
        range_v = float(max_v - min_v)
        print(range_v)
        if range_v > 1:
            bal[:,:,i] = bal[:,:,i] - min_v
            bal[:,:,i] =  bal[:,:,i]*255.0/(range_v)
        else:
            bal[:,:,i] = 0

    img = bal.astype(np.uint8)
    #img = cv2.cvtColor(img,cv2.COLOR_BGR2YCrCb)
    img = transform(img)
    #img = img[:,:,1]
    '''
    cv2.imshow("trans",img)
    cv2.waitKey(0)
