# Library imports
import cv2 as cv
import numpy as np

# Function for drawing the selected pitch points drawn on the image window
def redrawFrame(frame, imagePts):
    frame = frameCopy.copy()
    # Displaying coordinate counter
    cv.putText(frame, 'Select Pitch Point ' + str(len(imagePts)+1), (frame.shape[1]-200, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5 , (255,255,0), 1)
    
    # Displaying points selected
    for point in imagePts:
        # Skipping over points not present on pitch
        if (point == [-1,-1]):
            continue
        # Displaying a dot and the coordinates on the image window
        cv.putText(frame, str(point[0]) + ',' + str(point[1]), (point[0],point[1]), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv.circle(frame, (point[0],point[1]), 3, (0,255,255), -1)
    return frame

# Function for pitch point selection using mouse
def clickPitchPoints(event, x, y, flags, param):
    global imagePts
    global frame
    # Left button click to mark the next point
    if (event == cv.EVENT_LBUTTONDOWN):
        imagePts.append([x,y,1])
        frame = redrawFrame(frame, imagePts)
    # Right button click to remove the last point
    elif (event == cv.EVENT_RBUTTONDOWN) and (len(imagePts) > 0):
        imagePts.pop(len(imagePts)-1)
        frame = redrawFrame(frame, imagePts)
    # Any other mouse button to skip the next point (not present on image)
    elif (event == cv.EVENT_MBUTTONDOWN):
        imagePts.append([-1,-1])
        frame = redrawFrame(frame, imagePts)

# Function converting image point (from mouse event) to pitch point
def homography(event, x, y, flags, param):
    if (event == cv.EVENT_LBUTTONDOWN):
        result = h @ np.asarray([[x],[y],[1]])
        newX = result[0] / result[2]
        newY = result[1] / result[2]
        newZ = result[2] / result[2]
        print(np.asarray([[newX],[newY],[newZ]]))

# Real pitch dimensions - currently hardcoded but will later need user input
pitchLength = 105
pitchWidth = 68

# Mixture of gaussians background subtractor object
backSub = cv.createBackgroundSubtractorMOG2(800, 12, True)
backSub2 = cv.createBackgroundSubtractorMOG2(800, 16, True)
# backSub3 = cv.createBackgroundSubtractorMOG2(800, 12, True)

# Create a VideoCapture object and read from input file
cap = cv.VideoCapture('Datasets/Game1/First Half/0/output.h264') # 30s clip / 900 frames - camera 0
# cap = cv.VideoCapture('Datasets/Game1/First Half/1/output.h264') # 30s clip / 900 frames - camera 1

# Check if camera opened successfully
if (cap.isOpened()== False):
    print("Error opening video file")

# Read until video is completed
while(cap.isOpened()):

    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:

        # Obtains current frame number
        frameNumber = cap.get(cv.CAP_PROP_POS_FRAMES)

        # Select pitch points on first video frame
        if (frameNumber == 1):
            # Initialising selection window
            imagePts = []
            frameCopy = frame.copy()
            cv.namedWindow('Frame')
            cv.setMouseCallback('Frame', clickPitchPoints)
            cv.putText(frame, 'Select Pitch Point ' + str(len(imagePts)+1), (frame.shape[1]-200, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5 , (255,255,0), 1)
            
            # User selection of 29 key pitch poits
            while (len(imagePts) < 29):
                cv.imshow('Frame', frame)
                key = cv.waitKey(1) & 0xFF
            
            # Finding the homography matrix and resultant pitch mask after pitch points selected 
            else:
                # Removing mouse callback event
                cv.setMouseCallback('Frame', lambda *args : None)
                # cv.setMouseCallback('Frame', homography)

                # Homography calculation
                boxToSidelines = (pitchWidth-40.32)/2
                pitchPts = [[0, 0, 1], [0, boxToSidelines, 1], [0, boxToSidelines+11, 1],
                            [0, boxToSidelines+16.5, 1], [0, boxToSidelines+23.82, 1],
                            [0, boxToSidelines+29.32, 1], [0, boxToSidelines+40.32, 1],
                            [0, pitchWidth, 1], [5.5, boxToSidelines+11, 1],
                            [5.5, boxToSidelines+29.32, 1], [11, pitchWidth/2, 1],
                            [16.5, boxToSidelines, 1], [16.5, boxToSidelines+40.32, 1],
                            [pitchLength/2, 0, 1], [pitchLength/2, pitchWidth/2, 1],
                            [pitchLength/2, pitchWidth, 1], [pitchLength-16.5, boxToSidelines, 1],
                            [pitchLength-16.5, boxToSidelines+40.32, 1],
                            [pitchLength-11, pitchWidth/2, 1],
                            [pitchLength-5.5, boxToSidelines+11, 1],
                            [pitchLength-5.5, boxToSidelines+29.32, 1],
                            [pitchLength, 0, 1], [pitchLength,boxToSidelines, 1],
                            [pitchLength, boxToSidelines+11, 1],
                            [pitchLength, boxToSidelines+16.5, 1],
                            [pitchLength, boxToSidelines+23.82, 1],
                            [pitchLength, boxToSidelines+29.32, 1],
                            [pitchLength, boxToSidelines+40.32, 1],
                            [pitchLength, pitchWidth, 1]]
                imagePtsNew = []
                pitchPtsNew = []
                for i in range(len(imagePts)):
                    if (imagePts[i] != [-1,-1]):
                        imagePtsNew.append(imagePts[i])
                        pitchPtsNew.append(pitchPts[i])

                imageCoords = np.asarray(imagePtsNew)
                pitchCoords = np.asarray(pitchPtsNew)
                h, inliers = cv.findHomography(imageCoords, pitchCoords)

                # Pitch mask creation by iterating over each image pixel
                mask = np.zeros(frame.shape[:2], dtype='uint8')
                for i in range (mask.shape[0]):
                    for j in range (mask.shape[1]):
                        pitchPos = h @ np.asarray([[j],[i],[1]])
                        newX = pitchPos[0] / pitchPos[2]
                        newY = pitchPos[1] / pitchPos[2]
                        newZ = pitchPos[2] / pitchPos[2]
                        if (newX>0 and newX<pitchLength and newY>0 and newY<pitchWidth):
                            mask[i][j] = 1

        # Application of pitch mask
        frame = cv.bitwise_and(frame, frame, mask=mask)
        frame2 = frame.copy()
        # frame3 = frame.copy()

        fgMask = backSub.apply(frame)
        fgMask2 = backSub2.apply(frame2)
        # fgMask3 = backSub3.apply(frame3)

        ret, thresh = cv.threshold(fgMask,127,255,cv.THRESH_BINARY)
        ret2, thresh2 = cv.threshold(fgMask2,127,255,cv.THRESH_BINARY)
        # ret3, thresh3 = cv.threshold(fgMask3,127,255,cv.THRESH_BINARY)

        # 3x3 opening kernal was better
        kernel = np.ones((3,3),np.uint8)
        opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel)
        opening2 = cv.morphologyEx(thresh2, cv.MORPH_OPEN, kernel)
        # opening3 = cv.morphologyEx(thresh3, cv.MORPH_OPEN, kernel)

        contours, hierarchy = cv.findContours(opening, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        contours2, hierarchy2 = cv.findContours(opening2, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        # contours3, hierarchy3 = cv.findContours(opening3, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        
        cv.drawContours(frame, contours, -1, (0,255,0), 3)
        cv.drawContours(frame2, contours2, -1, (0,255,0), 3)
        # cv.drawContours(frame3, contours3, -1, (0,255,0), 3)

        for contour in contours:
            x,y,w,h = cv.boundingRect(contour)
            if (h > w) and (h > 10):
                cv.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)

        for contour in contours2:
            x,y,w,h = cv.boundingRect(contour)
            if (h > w) and (h > 10):
                cv.rectangle(frame2,(x,y),(x+w,y+h),(0,0,255),2)
        
        # for contour in contours3:
        #     x,y,w,h = cv.boundingRect(contour)
        #     if (h > w) and (h > 10):
        #         cv.rectangle(frame3,(x,y),(x+w,y+h),(0,0,255),2)

        # Current frame counter
        cv.rectangle(frame, (10, 2), (100,20), (255,255,255), -1)
        cv.putText(frame, str(frameNumber), (15, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))

        # Current frame counter
        cv.rectangle(frame2, (10, 2), (100,20), (255,255,255), -1)
        cv.putText(frame2, str(frameNumber), (15, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))

        # # Current frame counter
        # cv.rectangle(frame3, (10, 2), (100,20), (255,255,255), -1)
        # cv.putText(frame3, str(frameNumber), (15, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))


        # Display the resulting frame
        cv.imshow('Frame', frame)
        cv.imshow('Frame2', frame2)
        # cv.imshow('Frame3', frame3)

        # Pause at frames
        if (frameNumber == 90 or frameNumber == 270 or frameNumber == 450 or frameNumber == 720 or frameNumber == 900):
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