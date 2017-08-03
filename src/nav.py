import cv2
import imageio
import glob
import numpy as np

# TODO
# Shift in prespective transform
# Threshold map to be green

imgs = glob.glob("E:/dataset/*.png")
im = imageio.imread(imgs[400])
print(im.shape)

def nothing():
    pass

cv2.namedWindow('image')

# create trackbars for color change
cv2.createTrackbar('rL','image',0,255,nothing)
cv2.createTrackbar('rU','image',0,255,nothing)
cv2.createTrackbar('gL','image',0,255,nothing)
cv2.createTrackbar('gU','image',0,255,nothing)
cv2.createTrackbar('bL','image',0,255,nothing)
cv2.createTrackbar('bU','image',0,255,nothing)



for im in imgs:
    #lower = np.uint8([cv2.getTrackbarPos('bL','image'),cv2.getTrackbarPos('gL','image'),cv2.getTrackbarPos('rL','image')])
    #upper = np.uint8([cv2.getTrackbarPos('bU','image'),cv2.getTrackbarPos('gU','image'),cv2.getTrackbarPos('rU','image')])
    lower = np.uint8([0,86,0])
    upper = np.uint8([77,255,34])
    img = imageio.imread(im)
    mask = cv2.inRange(img, lower, upper)
    output = cv2.bitwise_and(img, img, mask=mask)

	# show the images
    cv2.imshow("images", cv2.resize(np.hstack([img, output]),None,fx=0.5,fy=0.5))
    cv2.waitKey(1)
