# importing libraries
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
 
# Output opencv version
print(cv.__version__)

def nothing(x):
    pass

backSub = cv.createBackgroundSubtractorMOG2()

# Create a VideoCapture object and read from input file
cap = cv.VideoCapture('Datasets/Game1/First Half/0/output.h264') # 30s clip / 900 frames
# cap = cv.VideoCapture('Datasets/Game1/First Half/1/output.h264') # 30s clip / 900 frames
# cap = cv.VideoCapture('Datasets/Game2/First Half/1/0165_2013-11-07 21_05_17.577813000.h264') # Game2 1st
# cap = cv.VideoCapture('Datasets/Game2/Second Half/1/1369_2013-11-07 22_05_29.585710000.h264') # Game2 2nd


# Check if camera opened successfully
if (cap.isOpened()== False):
    print("Error opening video file")

# Read until video is completed
while(cap.isOpened()):
     
# Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:

        # Current frame counter
        cv.rectangle(frame, (10, 2), (100,20), (255,255,255), -1)
        cv.putText(frame, str(cap.get(cv.CAP_PROP_POS_FRAMES)), (15, 15),
               cv.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))

        # # Display the resulting frame
        cv.imshow('Frame', frame)
        
        # Pause at frames 30, 60, 90
        frameNumber = cap.get(cv.CAP_PROP_POS_FRAMES)
        if (frameNumber == 30 or frameNumber == 90 or frameNumber == 270 or frameNumber == 450):

            # # Create a window
            # cv.namedWindow('image')

            # # Create trackbars for color change
            # # Hue is from 0-179 for Opencv
            # cv.createTrackbar('HMin', 'image', 0, 179, nothing)
            # cv.createTrackbar('SMin', 'image', 0, 255, nothing)
            # cv.createTrackbar('VMin', 'image', 0, 255, nothing)
            # cv.createTrackbar('HMax', 'image', 0, 179, nothing)
            # cv.createTrackbar('SMax', 'image', 0, 255, nothing)
            # cv.createTrackbar('VMax', 'image', 0, 255, nothing)

            # # Set default value for Max HSV trackbars
            # cv.setTrackbarPos('HMin', 'image', 40)
            # cv.setTrackbarPos('SMin', 'image', 40)
            # cv.setTrackbarPos('VMin', 'image', 30)
            # cv.setTrackbarPos('HMax', 'image', 70)
            # cv.setTrackbarPos('SMax', 'image', 255)
            # cv.setTrackbarPos('VMax', 'image', 255)

            # # Initialize HSV min/max values
            # hMin = sMin = vMin = hMax = sMax = vMax = 0
            # phMin = psMin = pvMin = phMax = psMax = pvMax = 0
            # while(1):
            #     # Get current positions of all trackbars
            #     hMin = cv.getTrackbarPos('HMin', 'image')
            #     sMin = cv.getTrackbarPos('SMin', 'image')
            #     vMin = cv.getTrackbarPos('VMin', 'image')
            #     hMax = cv.getTrackbarPos('HMax', 'image')
            #     sMax = cv.getTrackbarPos('SMax', 'image')
            #     vMax = cv.getTrackbarPos('VMax', 'image')

            #     # Set minimum and maximum HSV values to display
            #     lower = np.array([hMin, sMin, vMin])
            #     upper = np.array([hMax, sMax, vMax])

            #     # Convert to HSV format and color threshold
            #     hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
            #     mask = cv.inRange(hsv, lower, upper)
            #     result = cv.bitwise_and(frame, frame, mask=mask)
            #     # result2 = cv.rotate(result, cv.ROTATE_180)

            #     # Print if there is a change in HSV value
            #     if((phMin != hMin) | (psMin != sMin) | (pvMin != vMin) | (phMax != hMax) | (psMax != sMax) | (pvMax != vMax) ):
            #         print("(hMin = %d , sMin = %d, vMin = %d), (hMax = %d , sMax = %d, vMax = %d)" % (hMin , sMin , vMin, hMax, sMax , vMax))
            #         phMin = hMin
            #         psMin = sMin
            #         pvMin = vMin
            #         phMax = hMax
            #         psMax = sMax
            #         pvMax = vMax

            #     # Display result image
            #     cv.imshow('image', result)
            #     if cv.waitKey(10) & 0xFF == ord('q'):
            #         break
            

            # Initialize HSV min/max values
            hMin = sMin = vMin = 0
            hMax = sMax = vMax = 255
    
            sMin = vMin = 40
            hMin = 70

            # # hMin = sMin = 40 # sMin=150?
            # hMin = 40
            # sMin = 75
            # vMin = 30
            # hMax = 70
            # sMax = vMax = 255

            # Set minimum and maximum HSV values to display
            lower = np.array([hMin, sMin, vMin])
            upper = np.array([hMax, sMax, vMax])

            # Convert to HSV format and color threshold
            hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
            mask = cv.inRange(hsv, lower, upper)
            result = cv.bitwise_and(frame, frame, mask=mask)
            # result = cv.bitwise_not(result)
            
            # # Select ROI
            # roi = cv.selectROI("select the area", result)

            # # Opening
            # kernel = np.ones((3,3),np.uint8)
            # opening = cv.morphologyEx(result, cv.MORPH_OPEN, kernel)
            # kernel2 = np.ones((5,5),np.uint8)
            # opening2 = cv.morphologyEx(result, cv.MORPH_OPEN, kernel2)
            # kernel3 = np.ones((7,7),np.uint8)
            # opening3 = cv.morphologyEx(result, cv.MORPH_OPEN, kernel3)

            bgr = cv.cvtColor(result, cv.COLOR_HSV2BGR)
            grey = cv.cvtColor(bgr, cv.COLOR_BGR2GRAY)

            image = grey
            threshold1 = 0
            threshold2 = 1000
            apertureSize = 5
            L2gradient = True
            canny = cv.Canny(image=image, threshold1=threshold1, threshold2=threshold2, apertureSize=apertureSize, L2gradient=L2gradient)

            # Probabilistic Hough
            # linesP = cv.HoughLinesP(image=canny, rho=1, theta=np.pi/180, threshold=50, minLineLength=200, maxLineGap=20)
            linesP = cv.HoughLinesP(image=canny, rho=1, theta=np.pi/180, threshold=50, minLineLength=175, maxLineGap=20)
            
            if linesP is not None:
                for i in range(0, len(linesP)):
                    l = linesP[i][0]
                    cv.line(result, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv.LINE_AA)

            cv.imshow('image', result)
            # cv.imshow('opening', opening)
            # cv.imshow('opening2', opening2)
            # cv.imshow('opening3', opening3)
            # cv.imshow('grey', grey)
            cv.imshow('canny', canny)
            cv.waitKey(0)

        # Press Q on keyboard to exit
        if cv.waitKey(25) & 0xFF == ord('q'):
            break
 
# Break the loop
    else:
        break
 
# When everything done, release the video capture object
cap.release()
 
# Closes all the frames
cv.destroyAllWindows()
