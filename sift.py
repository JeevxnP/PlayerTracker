# importing libraries
import cv2 as cv
import numpy as np

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

        # Display the resulting frame
        cv.imshow('Frame', frame)
        
        # Pause at frames
        if (frameNumber == 90 or frameNumber == 270 or frameNumber == 450 or frameNumber == 720 or frameNumber == 900):
            
            # Create SIFT feature extractor
            sift = cv.xfeatures2d.SIFT_create()

            # # Convert image to greyscale
            # grey = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            # Detect features from the image
            keypoints, descriptors = sift.detectAndCompute(frame, None)

            # Draw the detected key points
            sift_image = cv.drawKeypoints(frame, keypoints, frame)

            print(keypoints)
            print(descriptors)

            # Show the image
            cv.imshow('SIFT Features', sift_image)

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