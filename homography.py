# Library imports
import cv2 as cv
import numpy as np

# Function for pitch point selection using mouse
def clickPitchPoints(event, x, y, flags, param):
    global pitchPts
    if (event == cv.EVENT_LBUTTONDOWN):
        pitchPts.append([x,y])
        print(pitchPts)
        # displaying the coordinates on the image window
        font = cv.FONT_HERSHEY_SIMPLEX
        cv.putText(frame, str(x) + ',' +
                    str(y), (x,y), font,
                    0.5, (255, 255, 0), 1)
        cv.imshow('Frame', frame)
    elif (event == cv.EVENT_RBUTTONDOWN) and (len(pitchPts) > 0):
        pitchPts.pop(len(pitchPts)-1)
        print(pitchPts)
    elif (event == cv.EVENT_MBUTTONDOWN):
        pitchPts.append([-1,-1])
        print(pitchPts)

# Function removing pitch point selection functionality for mouse
def noMouseCallback(event, x, y, flags, param):
    pass

# Mixture of gaussians background subtractor object
backSub = cv.createBackgroundSubtractorMOG2()

# Create a VideoCapture object and read from input file
cap = cv.VideoCapture('Datasets/Game1/First Half/1/output.h264') # 30s clip / 900 frames

# Check if camera opened successfully
if (cap.isOpened()== False):
    print("Error opening video file")

# Read until video is completed
while(cap.isOpened()):

    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:

        # Current frame counter
        frameNumber = cap.get(cv.CAP_PROP_POS_FRAMES)
        cv.rectangle(frame, (10, 2), (100,20), (255,255,255), -1)
        cv.putText(frame, str(frameNumber), (15, 15),
               cv.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))

        # Select pitch points on first video frame
        if (frameNumber == 1):
            pitchPts = []
            cv.namedWindow('Frame')
            cv.setMouseCallback('Frame', clickPitchPoints)
            while (len(pitchPts) <= 5):
                cv.imshow('Frame', frame)
                key = cv.waitKey(1) & 0xFF
            else:
                cv.setMouseCallback('Frame', noMouseCallback)
        else:
            # Display the resulting frame
            cv.imshow('Frame', frame)

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