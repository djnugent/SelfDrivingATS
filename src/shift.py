import cv2
import imageio
import numpy as np
import glob


imgs = glob.glob("E:/dataset/*.png")


def nothing():
    pass

cv2.namedWindow('image')

# create trackbars for color change
cv2.createTrackbar('shift','image',50,100,nothing)
cv2.createTrackbar('rotation','image',0,25,nothing)



def perspective_transform(img):
    # Choose an offset from image corners to plot detected corners
    lane_width = 130 #85
    apex_width = 80 #230
    horizon0 = 415 #485
    horizon1 = 0 #610

    # Grab the image shape
    h,w,c = img.shape
    # Specify the transform
    src = np.float32([[0,h],
                    [w,h],
                    [int(w/2 - apex_width/2),horizon0],
                    [int(w/2 + apex_width/2),horizon0]])
    dst = np.float32([[int(w/2 - lane_width/2), h],
                      [int(w/2 + lane_width/2), h],
                      [int(w/2 - lane_width/2), horizon1],
                      [int(w/2 + lane_width/2), horizon1]])

    # Given src and dst points, calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    inv_M = cv2.getPerspectiveTransform(dst, src)
    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(img, M, (w,h))
    return warped, M, inv_M


def warpPoints(pnts,M):
    new_pnts = []
    for pnt in pnts:
        new_pnts.append(np.matmul(M,np.array([pnt[0],pnt[1],1])))
    return new_pnts


def shiftPoints(pnts,x,y):
    M = np.float32([[1,0,x],[0,1,y],[0,0,1]])
    return cv2.perspectiveTransform(np.array([pnts]),M)[0]

def rotatePoints(pnts,img_shape,angle):
    w,h,c = img_shape
    center = int(w/2),h
    M = cv2.getRotationMatrix2D(center,angle,1.0)
    M = np.concatenate((M, np.array([[0,0,1]])), axis=0)
    return cv2.perspectiveTransform(np.array([pnts]),M)[0]


def augment(img, rotation, shift, M, inv_M):
    h,w,c = img.shape


    # Warp points birds eye view
    src = np.array([[200,500],[555,500],[80,550],[577,550]], dtype='float32')
    warped = cv2.perspectiveTransform(np.array([src]),M)[0]

    # shift points
    x = shift
    y = 0
    trans_M = np.float32([[1,0,x],[0,1,y],[0,0,1]])
    warped_shifted = cv2.perspectiveTransform(np.array([warped]),trans_M)[0]
    print(warped_shifted)


    # Rotate points
    center = int(w/2),h
    rot_M = cv2.getRotationMatrix2D(center,rotation,1.0)
    rot_M = np.concatenate((rot_M, np.array([[0,0,1]])), axis=0)
    warped_rot = cv2.perspectiveTransform(np.array([warped_shifted]),rot_M)[0]
    print(warped_rot)

    # Warp back to regular perspective
    warped_final = cv2.perspectiveTransform(np.array([warped_rot]),inv_M)[0]

    # Apply transfom to image
    final_M = cv2.getPerspectiveTransform(src, warped_final)
    # Warp the image using OpenCV warpPerspective()
    result = cv2.warpPerspective(img, final_M, (w,h))

    return result

for img in imgs:
    im = imageio.imread(img)

    warped, M,inv_M = perspective_transform(im)
    shift = cv2.getTrackbarPos('shift','image')
    rot = cv2.getTrackbarPos('rotation','image')
    result = augment(im,rot,shift,M,inv_M)
    #cv2.imshow("orig",im)
    cv2.imshow("result",result)
    cv2.waitKey(1)
