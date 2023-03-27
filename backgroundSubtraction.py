# importing libraries
import cv2 as cv
import numpy as np
 
# Output opencv version
print(cv.__version__)

# backSub = cv.createBackgroundSubtractorMOG2()
# backSub2 = cv.createBackgroundSubtractorMOG2(500, 16, True) # default params
backSub = cv.createBackgroundSubtractorMOG2(800, 12, True)
# backSub2 = cv.createBackgroundSubtractorMOG2(800, 16, True)
# backSub2 = cv.createBackgroundSubtractorMOG2(500, 16, False)
# backSub3 = cv.createBackgroundSubtractorMOG2(800, 16, True)

# Create a VideoCapture object and read from input file
cap = cv.VideoCapture('Datasets/Game1/First Half/1/output.h264') # 30s clip / 900 frames
# cap2 = cv.VideoCapture('Datasets/Game1/First Half/1/output.h264') # 30s clip / 900 frames
# cap = cv.VideoCapture('Datasets/Game1/First Half/1/0056_2013-11-03 18_01_14.248366000.h264') # Game1 1st
# cap = cv.VideoCapture('Datasets/Game1/First Half/0/0056_2013-11-03 18_01_14.248366000.h264') # Game1 1st
# cap = cv.VideoCapture('Datasets/Game1/Second Half/1/1323_2013-11-03 19_04_35.246651000.h264') # Game1 2nd
# cap = cv.VideoCapture('Datasets/Game2/First Half/1/0165_2013-11-07 21_05_17.577813000.h264') # Game2 1st
# cap = cv.VideoCapture('Datasets/Game2/Second Half/1/1369_2013-11-07 22_05_29.585710000.h264') # Game2 2nd
 
# Check if camera opened successfully
if (cap.isOpened()== False):
    print("Error opening video file")

# Read until video is completed
while(cap.isOpened()):
     
# Capture frame-by-frame
    ret, frame = cap.read()
    # ret2, frame2 = cap2.read()
    if ret == True:

        # Output image dimensions
        # print(frame.shape)

        # # Initialize HSV min/max values
        # hMin = sMin = vMin = 0
        # hMax = sMax = vMax = 255

        # # Set minimum and maximum HSV values to display
        # lower = np.array([hMin, sMin, vMin])
        # upper = np.array([hMax, sMax, vMax])

        # # Convert to HSV format and color threshold
        # hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        # mask = cv.inRange(hsv, lower, upper)
        # result = cv.bitwise_and(frame2, frame2, mask=mask)

        fgMask = backSub.apply(frame)
        # fgMask2 = backSub2.apply(frame2)
        # fgMask2 = backSub.apply(image=frame, learningRate=0.02)

        ret, thresh = cv.threshold(fgMask,127,255,cv.THRESH_BINARY)
        # ret2, thresh2 = cv.threshold(fgMask2,127,255,cv.THRESH_BINARY)

        # Current frame counter
        cv.rectangle(frame, (10, 2), (100,20), (255,255,255), -1)
        cv.putText(frame, str(cap.get(cv.CAP_PROP_POS_FRAMES)), (15, 15),
               cv.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
        # # Current frame counter
        # cv.rectangle(frame2, (10, 2), (100,20), (255,255,255), -1)
        # cv.putText(frame2, str(cap.get(cv.CAP_PROP_POS_FRAMES)), (15, 15),
        #        cv.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))

        # 3x3 opening kernal was better
        kernel = np.ones((3,3),np.uint8)
        opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel)
        # opening2 = cv.morphologyEx(thresh, cv.MORPH_ERODE, kernel, iterations=2)
        # opening2 = cv.morphologyEx(opening2, cv.MORPH_DILATE, kernel, iterations=3)
        # opening2 = cv.morphologyEx(thresh2, cv.MORPH_OPEN, kernel)

        contours, hierarchy = cv.findContours(opening, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        # contours2, hierarchy2 = cv.findContours(opening2, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        cv.drawContours(frame, contours, -1, (0,255,0), 3)
        # cv.drawContours(frame2, contours2, -1, (0,255,0), 3)

        for contour in contours:
            x,y,w,h = cv.boundingRect(contour)
            if (h > w) and (h > 10):
                cv.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)

        # for contour in contours2:
        #     x,y,w,h = cv.boundingRect(contour)
        #     if (h > w) and (h > 10):
        #         # cv.rectangle(frame2,(x,y),(x+w,y+h),(0,0,255),2)

        # # Display the resulting frame
        cv.imshow('Frame', frame)
        # cv.imshow('Frame2', frame2)
        # cv.imshow('Blur', blur)
        # cv.imshow('FG Mask', fgMask)
        # cv.imshow('Thresholded', thresh)
        # cv.imshow('Opened', opening)
        # cv.imshow('Opened2', opening2)
        
        frameNumber = cap.get(cv.CAP_PROP_POS_FRAMES)

        # Pause at frames
        if (frameNumber == 10 or frameNumber == 20 or frameNumber == 30 or frameNumber == 40 or frameNumber == 50):
            cv.waitKey(0)

        # Pause at frames
        if (frameNumber == 90 or frameNumber == 270 or frameNumber == 450 or frameNumber == 720 or frameNumber == 900):
            # print("Contours length is:" , len(contours))
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
