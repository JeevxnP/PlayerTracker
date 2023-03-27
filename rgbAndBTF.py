# importing libraries
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def cumulativeBrightness(c):
    # Initialisation
    cumulativeHists = np.zeros((3,256,1), dtype='float32')
    # inverseCumulativeHists = [None]*3

    # Amalgamation of image histograms
    for image in c:
        for i in range(3):
            cumulativeHists[i] += cv.calcHist([image],[i],None,[256],[0,256])

    for i in range(3):
        # Normalise and make cumulative histogram
        cumulativeHists[i] /= cumulativeHists[i].sum()
        cumulativeHists[i] = np.asarray([np.cumsum(cumulativeHists[i])]).T

        # WE WANT INVERTED NOT INVERSE!
        # # Make inverse cumulative histogram
        # inverseCumulativeHists[i] = np.roll((1-cumulativeHists[i]), 1)
        # inverseCumulativeHists[i][0] = 1

    return cumulativeHists

def invertedIndex(histogram, pixelPercentage):
    # Eliminates invalid pixel percentage
    if (pixelPercentage<0) or (pixelPercentage>1):
        return -1
    
    # Finds and returns index which pixel percentage lies in
    for i in range (len(histogram)):
        if (pixelPercentage<=histogram[i]):
            return i
    
    # Invalid state (shouldn't reach here)
    return -1
        
def cbtf(collection1, collection2):
    cumHist1 = cumulativeBrightness(collection1)
    cumHist2 = cumulativeBrightness(collection2)

    newHists = np.zeros((len(collection2),3,256,1), dtype='float32')
    for i,image in enumerate(collection2):
        for j in range(3):
            histogram = cv.calcHist([image],[j],None,[256],[0,256])
            for bin,count in enumerate(histogram):
                newIndex = invertedIndex(cumHist1[j], cumHist2[j][bin])
                newHists[i][j][newIndex] += count[0]
    
    return cumHist1, cumHist2, newHists

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

        frame2 = frame.copy()
        frame2 = np.where((255 - frame2) < 15,255,frame2+15)

        # Current frame counter
        frameNumber = cap.get(cv.CAP_PROP_POS_FRAMES)
        cv.rectangle(frame, (10, 2), (100,20), (255,255,255), -1)
        cv.putText(frame, str(frameNumber), (15, 15),
               cv.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))

        # Display the resulting frame
        cv.imshow('Frame', frame)
        cv.imshow('Frame2', frame2)
        
        # Pause at frames
        if (frameNumber == 90 or frameNumber == 270 or frameNumber == 450 or frameNumber == 720 or frameNumber == 900):

            # # Rectangle top-left and bottom-right coords
            # point1 = [100, 100]
            # point2 = [300, 400]

            # mask = np.zeros(frame.shape[:2], dtype='uint8')
            # cv.rectangle(mask, (point1[0],point1[1]), (point2[0],point2[1]), 255, -1)
            # cv.imshow('Mask', mask)

            # maskedImage = cv.bitwise_and(frame, frame, mask=mask)
            # cv.imshow('Masked Image', maskedImage)

            # histr3 = cumulativeBrightness([frame, frame])
            # cumHist1, cumHist2, newHists = cbtf([frame], [frame])
            cumHist1, cumHist2, newHists = cbtf([frame], [frame2])

            # RGB Histograms
            color = ('b','g','r')
            fig, ax = plt.subplots(5, 1)
            plt.subplots_adjust(left=0.2,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.2,
                    hspace=0.6)
            for i,col in enumerate(color):
                # Full image RGB histogram
                ax[0].set_title("Full Image RGB Histogram")
                ax[0].set_xlabel("Bins")
                ax[0].set_ylabel("# of Pixels")
                ax[0].set_xlim([0,256])
                histr1 = cv.calcHist([frame],[i],None,[256],[0,256])
                ax[0].plot(histr1, color=col)

                # Masked image RGB histogram
                ax[1].set_title("Full Image Cumulative RGB Histogram")
                ax[1].set_xlabel("Bins")
                ax[1].set_ylabel("# of Pixels")
                ax[1].set_xlim([0,256])
                
                histr2 = cv.calcHist([frame2],[i],None,[256],[0,256])
                ax[1].plot(histr2, color=col)

                ax[2].plot(newHists[0][i], color=col)

                ax[3].plot(cumHist1[i], color=col)
                print(cumHist1[i][0])
                print(cumHist2[i][16])
                ax[4].plot(cumHist2[i], color=col)

                # # Masked image RGB histogram
                # ax[1].set_title("Masked Image RGB Histogram")
                # ax[1].set_xlabel("Bins")
                # ax[1].set_ylabel("# of Pixels")
                # ax[1].set_xlim([0,256])
                # histr2 = cv.calcHist([maskedImage],[i],None,[256],[0,256])
                # ax[1].plot(histr2, color=col)

            plt.show()

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