# Library imports
import cv2 as cv
import numpy as np

# Function converting image point (from mouse event) to pitch point
def homography(event, x, y, flags, param):
    if (event == cv.EVENT_LBUTTONDOWN):
        # result = h @ np.asarray([[x],[y],[1]])
        # newX = result[0] / result[2]
        # newY = result[1] / result[2]
        # newZ = result[2] / result[2]
        # print(np.asarray([[newX],[newY],[newZ]]))
        print(np.asarray([x, y]))


minimapCoords = np.asarray([[25, 9, 1], [25, 77, 1], [25, 131, 1], [25, 157, 1],
            [25, 193, 1], [25, 219, 1], [25, 273, 1], [25, 341, 1], [51, 131, 1],
            [51, 219, 1], [79, 175, 1], [105, 77, 1], [105, 273, 1], [282, 9, 1],
            [282, 175, 1], [282, 341, 1], [459, 77, 1], [459, 273, 1], [486, 175, 1],
            [513, 131, 1], [513, 219, 1], [539, 9, 1], [539,77, 1], [539, 131, 1],
            [539, 157, 1], [539, 193, 1], [539, 219, 1], [539, 273, 1], [539, 341, 1]])


minimap = cv.imread('Pitch Minimap2.jpg')
print(minimap.shape)
minimap = cv.resize(minimap, None, fx=2, fy=2)
print(minimap.shape)

cv.imshow('Minimap', minimap)

cv.setMouseCallback('Minimap', homography)

cv.waitKey(0)
cv.destroyAllWindows()