# Library imports
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Function for drawing the selected pitch points drawn on the image window
def redrawFrame(frame, minimap, imagePts):
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
    
    minimap = minimapCopy.copy()
    cv.circle(minimap, (minimapPts[len(imagePts)][0],minimapPts[len(imagePts)][1]), 3, (0,255,255), -1)

    return frame, minimap

# Function for pitch point selection using mouse
def clickPitchPoints(event, x, y, flags, param):
    global imagePts
    global frame
    global minimap
    # Left button click to mark the next point
    if (event == cv.EVENT_LBUTTONDOWN):
        imagePts.append([x,y,1])
        if (len(imagePts) < 29):
            frame, minimap = redrawFrame(frame, minimap, imagePts)
    # Right button click to remove the last point
    elif (event == cv.EVENT_RBUTTONDOWN) and (len(imagePts) > 0):
        imagePts.pop(len(imagePts)-1)
        frame, minimap = redrawFrame(frame, minimap, imagePts)
    # Any other mouse button to skip the next point (not present on image)
    elif (event == cv.EVENT_MBUTTONDOWN):
        imagePts.append([-1,-1])
        if (len(imagePts) < 29):
            frame, minimap = redrawFrame(frame, minimap, imagePts)

def homography(x,y):
    # Image coordinate to pitch coordinate
    pitchPoint = hImageToPitch @ np.asarray([[x],[y],[1]])
    newX = pitchPoint[0][0] / pitchPoint[2][0]
    newY = pitchPoint[1][0] / pitchPoint[2][0]
    newZ = pitchPoint[2][0] / pitchPoint[2][0]
    pitchPoint = np.asarray([[newX],[newY],[newZ]])

    # Pitch coordinate to minimap coordinate
    minimapPoint = hPitchToMinimap @ pitchPoint
    newX = minimapPoint[0][0] / minimapPoint[2][0]
    newY = minimapPoint[1][0] / minimapPoint[2][0]
    newZ = minimapPoint[2][0] / minimapPoint[2][0]
    minimapPoint = np.asarray([[newX],[newY],[newZ]])
    # Draws a point on the minimap
    cv.circle(minimap, (int(minimapPoint[0][0]),int(minimapPoint[1][0])), 3, (0,255,255), -1)

# Function converting image point (from mouse event) to pitch point
def imageToMinimap(event, x, y, flags, param):
    if (event == cv.EVENT_LBUTTONDOWN):
        homography(x,y)

# Function to output image coordinate of a mouse left button event
def coord(event, x, y, flags, param):
    if (event == cv.EVENT_LBUTTONDOWN):
        print(np.asarray([x, y]))

# Function to calculate whether two rectangles overlap
def calculateRectangleOverlap(tl1, br1, tl2, br2):
     
    # If one rectangle is on left side of other
    if ((tl1[0]<=tl2[0] and br1[0]<=tl2[0]) or (tl1[0]>=br2[0] and br1[0]>=br2[0])):
        return False
 
    # If one rectangle is above other
    if ((tl1[1]<=tl2[1] and br1[1]<=tl2[1]) or (tl1[1]>=br2[1] and br1[1]>=br2[1])):
        return False
 
    # Otherwise they overlap
    return True

# Function for merging overlapping bounding boxes
def mergeOverlappingBoxes(BBs):
    i = 0
    while (i < len(BBs)-1):
        for j in range (i+1,len(BBs)):

            # Calculates top-left and bottom-right points of rectangles
            tl1 = [BBs[i][0],BBs[i][1]]
            br1 = [BBs[i][0]+BBs[i][2],BBs[i][1]+BBs[i][3]]
            tl2 = [BBs[j][0],BBs[j][1]]
            br2 = [BBs[j][0]+BBs[j][2],BBs[j][1]+BBs[j][3]]

            if (calculateRectangleOverlap(tl1, br1, tl2, br2)):
                area1 = BBs[i][2]*BBs[i][3]
                area2 = BBs[j][2]*BBs[j][3]
                ratio = area1/area2
                if not ((ratio>0.5) and (ratio<2)):
                    BBs[i] = cv.boundingRect(np.asarray([tl1, br1, tl2, br2]))
                    BBs.pop(j)
                    i = -1
                    break
        i+=1
    
    return BBs

def labValues(event, x, y, flags, param):
    if (event == cv.EVENT_LBUTTONDOWN):
        print("-----")
        print(np.asarray([x, y]))
        print(labImage[y][x])
        print("-----")

def cumulativeBrightness(c, b):
    # Initialisation
    cumulativeHists = np.zeros((3,b,1), dtype='float32')

    # Amalgamation of image histograms
    for image in c:
        for i in range(3):
            cumulativeHists[i] += cv.calcHist([image],[i],None,[b],[0,256])

    for i in range(3):
        # Normalise and make cumulative histogram
        cumulativeHists[i] /= cumulativeHists[i].sum()
        cumulativeHists[i] = np.asarray([np.cumsum(cumulativeHists[i])]).T

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

def btf(image1, image2, bins):
    newHists = np.zeros((3,bins,1), dtype='float32')

    for i in range(3):
        histogram1 = cv.calcHist([image1],[i],None,[bins],[0,256])
        histogram1 /= histogram1.sum()
        cumHist1 = np.asarray([np.cumsum(histogram1)]).T

        histogram2 = cv.calcHist([image2],[i],None,[bins],[0,256])
        histogram2 /= histogram2.sum()
        cumHist2 = np.asarray([np.cumsum(histogram2)]).T

        for bin,count in enumerate(histogram2):
            newIndex = invertedIndex(cumHist1, cumHist2[bin])
            newHists[i][newIndex] += count[0]

    return newHists

def btfHybrid(cumulativeHist, image, bins):
    newHists = np.zeros((3,bins,1), dtype='float32')

    for i in range(3):
        histogram2 = cv.calcHist([image],[i],None,[bins],[0,256])
        histogram2 /= histogram2.sum()
        cumHist2 = np.asarray([np.cumsum(histogram2)]).T

        for bin,count in enumerate(histogram2):
            newIndex = invertedIndex(cumulativeHist[i], cumHist2[bin])
            newHists[i][newIndex] += count[0]

    return newHists

def cbtf(collection1, collection2, bins):
    cumHist1 = cumulativeBrightness(collection1, bins)
    cumHist2 = cumulativeBrightness(collection2, bins)

    newHists = np.zeros((len(collection2),3,bins,1), dtype='float32')
    for i,image in enumerate(collection2):
        for j in range(3):
            histogram = cv.calcHist([image],[j],None,[bins],[0,256])
            for bin,count in enumerate(histogram):
                newIndex = invertedIndex(cumHist1[j], cumHist2[j][bin])
                newHists[i][j][newIndex] += count[0]
            newHists[i][j] /= newHists[i][j].sum()
    
    return newHists


# Real pitch dimensions - currently hardcoded but will later need user input
pitchLength = 105
pitchWidth = 68

# Key pitch points
boxToSidelines = (pitchWidth-40.32)/2
pitchPts = [[0, 0, 1], [0, boxToSidelines, 1], [0, boxToSidelines+11, 1],
            [0, boxToSidelines+16.5, 1], [0, boxToSidelines+23.82, 1],
            [0, boxToSidelines+29.32, 1], [0, boxToSidelines+40.32, 1],
            [0, pitchWidth, 1], [5.5, boxToSidelines+11, 1],
            [5.5, boxToSidelines+29.32, 1], [11, pitchWidth/2, 1],
            [16.5, boxToSidelines, 1], [16.5, boxToSidelines+40.32, 1],
            [pitchLength/2, 0, 1], [pitchLength/2, pitchWidth/2, 1],
            [pitchLength/2, pitchWidth, 1], [pitchLength-16.5, boxToSidelines, 1],
            [pitchLength-16.5, boxToSidelines+40.32, 1], [pitchLength-11, pitchWidth/2, 1],
            [pitchLength-5.5, boxToSidelines+11, 1], [pitchLength-5.5, boxToSidelines+29.32, 1],
            [pitchLength, 0, 1], [pitchLength,boxToSidelines, 1],
            [pitchLength, boxToSidelines+11, 1], [pitchLength, boxToSidelines+16.5, 1],
            [pitchLength, boxToSidelines+23.82, 1], [pitchLength, boxToSidelines+29.32, 1],
            [pitchLength, boxToSidelines+40.32, 1], [pitchLength, pitchWidth, 1]]

# Key minimap points
minimapPts = [[25, 9, 1], [25, 77, 1], [25, 131, 1], [25, 157, 1], [25, 193, 1],
            [25, 219, 1], [25, 273, 1], [25, 341, 1], [51, 131, 1], [51, 219, 1],
            [79, 175, 1], [105, 77, 1], [105, 273, 1], [282, 9, 1], [282, 175, 1], 
            [282, 341, 1], [459, 77, 1], [459, 273, 1], [486, 175, 1], [513, 131, 1], 
            [513, 219, 1], [539, 9, 1], [539,77, 1], [539, 131, 1], [539, 157, 1], 
            [539, 193, 1], [539, 219, 1], [539, 273, 1], [539, 341, 1]]

# Mixture of gaussians background subtractor object
backSub = cv.createBackgroundSubtractorMOG2(800, 12, True)

# Create a VideoCapture object and read from input file
# cap = cv.VideoCapture('Datasets/Game1/First Half/0/output.h264') # 30s clip / 900 frames - camera 0
# cap = cv.VideoCapture('Datasets/Game1/First Half/0/output2.h264') # 30s clip / 900 frames - camera 0
# cap = cv.VideoCapture('Datasets/Game1/First Half/0/longerOutput.h264') # 60s clip / 1800 frames - camera 0
# cap = cv.VideoCapture('Datasets/Game1/First Half/1/output.h264') # 30s clip / 900 frames - camera 1
# cap = cv.VideoCapture('Datasets/Game1/First Half/1/output2.h264') # 30s clip / 900 frames - camera 1
cap = cv.VideoCapture('Datasets/Game1/First Half/1/longerOutput.h264') # 60s clip / 1800 frames - camera 1

# 300/400/500 for noise - 500 for undetected player
# 1200/1300/1400 for noise - 1400 for undetected player

# Pitch minimap
minimap = cv.imread('Pitch Minimap2.jpg')
minimap = cv.resize(minimap, None, fx=2, fy=2)

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
            # # Initialising selection window
            imagePts = []
            frameCopy = frame.copy()
            minimapCopy = minimap.copy()
            # cv.namedWindow('Frame')
            # cv.setMouseCallback('Frame', clickPitchPoints)
            # frame, minimap = redrawFrame(frame, minimap, imagePts)

            # # User selection of 29 key pitch points
            while (len(imagePts) < 29):
                ##### Temporary shortcut - REMOVE EVENTUALLY #####
                # Camera 0
                # imagePts = [[819, 161, 1], [740, 172, 1], [654, 184, 1], [605, 193, 1], [528, 204, 1], [470, 216, 1], [317, 247, 1], [55, 302, 1], [701, 191, 1], [519, 226, 1], [679, 215, 1], [896, 192, 1], [480, 296, 1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1]]
                # Camera 1
                imagePts = [[22, 147, 1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [140, 168, 1], [-1, -1], [640, 110, 1], [674, 196, 1], [872, 779, 1], [1139, 116, 1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [1226, 87, 1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1]]
                cv.imshow('Frame', frame)
                cv.imshow('Minimap', minimap)
                key = cv.waitKey(1) & 0xFF
            
            # Finding the homography matrix and resultant pitch mask after pitch points selected 
            else:
                # Need to stop imagePts from being full [-1, -1] from getting here
                # And also if there are an insufficient number of points selected in the frame
                # findHomography needs at least 4 corresponding points

                # Removing mouse callback event
                # cv.setMouseCallback('Frame', lambda *args : None)
                cv.setMouseCallback('Frame', imageToMinimap)
                # cv.setMouseCallback('Frame', coord)

                # print(imagePts)

                # Homography calculation
                imagePtsNew = []
                pitchPtsNew = []
                for i in range (len(imagePts)):
                    if (imagePts[i] != [-1,-1]):
                        imagePtsNew.append(imagePts[i])
                        pitchPtsNew.append(pitchPts[i])

                # Homography matrices calculation
                hImageToPitch, inliersImageToPitch = cv.findHomography(np.asarray(imagePtsNew), np.asarray(pitchPtsNew))
                hPitchToMinimap, inliersPitchToMinimap = cv.findHomography(np.asarray(pitchPts), np.asarray(minimapPts))

                # Pitch mask creation by iterating over each image pixel
                pitchMask = np.zeros(frame.shape[:2], dtype='uint8')
                for i in range (pitchMask.shape[0]):
                    for j in range (pitchMask.shape[1]):
                        pitchPos = hImageToPitch @ np.asarray([[j],[i],[1]])
                        newX = pitchPos[0][0] / pitchPos[2][0]
                        newY = pitchPos[1][0] / pitchPos[2][0]
                        newZ = pitchPos[2][0] / pitchPos[2][0]
                        if (newX>0 and newX<pitchLength and newY>0 and newY<pitchWidth):
                            pitchMask[i][j] = 1

        # Application of pitch mask
        frame = cv.bitwise_and(frame, frame, mask=pitchMask)

        # Copy of frame for k-means
        frameCopy = frame.copy()

        # Background subtraction on frame
        fgMask = backSub.apply(frame)

        # Thresholds detected shadows out
        ret, thresh = cv.threshold(fgMask,127,255,cv.THRESH_BINARY)

        # Opening morphological operation to elimate noise
        # 3x3 opening kernal was better
        kernel = np.ones((3,3),np.uint8)
        opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel)

        # Find and draw contours on frame + finding suitable bounding boxes
        contours, hierarchy = cv.findContours(opening, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        # cv.drawContours(frame, usedContours, -1, (0,255,0), 3)
        
        boundingBoxes = []

        # Might be better to use contour instead of bounding box for classification features
        # usedContours = []
        for contour in contours:
            x,y,w,h = cv.boundingRect(contour)
            if (h > w) and (h > 10) and (w > 10):
                boundingBoxes.append((x, y, w, h))
                # usedContours.append(contour)
        
        boundingBoxes = mergeOverlappingBoxes(boundingBoxes)
        
        # players = [6,7,8,11,13]   # Comparison
        # players = [6,7,8,1,0]   # Comparison
        # players = [0,2,3,5,8,10,13,14,15]  # Black/Blue team
        players = [1,4,7,9,11,12] # White/Red team
        # players = [6,6] # Referee

        # cv.drawContours(frame, usedContours, -1, (0,255,0), 3)

        colour = ('b','g','r')
        bins = 256
        featureMatrix = np.zeros((len(boundingBoxes),(3*bins)), dtype='float32')

        # Drawing bounding boxes on frame with numbering
        croppedImages = []
        for i in range (len(boundingBoxes)):
            x1 = boundingBoxes[i][0]
            y1 = boundingBoxes[i][1]
            x2 = boundingBoxes[i][0] + boundingBoxes[i][2]
            y2 = boundingBoxes[i][1] + boundingBoxes[i][3]
            cv.rectangle(frame,(boundingBoxes[i][0],boundingBoxes[i][1]),(boundingBoxes[i][0]+boundingBoxes[i][2],boundingBoxes[i][1]+boundingBoxes[i][3]),(0,0,255),1)
            cv.putText(frame, str(i), (boundingBoxes[i][0], boundingBoxes[i][1]), cv.FONT_HERSHEY_SIMPLEX, 0.5 , (255,255,0))

            # croppedImages.append(frameCopy[y1:y2, x1:x2].copy())

            croppedImage = frameCopy[y1:y2, x1:x2]
            # hsv = cv.cvtColor(croppedImage, cv.COLOR_BGR2HSV)
            lab = cv.cvtColor(croppedImage, cv.COLOR_BGR2LAB)
            features = np.asarray([])
            for j,col in enumerate(colour):
                # if j == 0:
                #     continue
                histogram = cv.calcHist([lab],[j],None,[bins],[0,256])
                histogram /= histogram.sum()
                # histogram -= histogram.mean()
                # histogram /= histogram.std()
                features = np.concatenate((features, histogram.T[0]))
            featureMatrix[i] = features
        
        clusters = 3
        startingFrame = 95
        if (len(boundingBoxes) >= clusters) and (frameNumber>=startingFrame):
            # kmeans = KMeans(n_clusters=3, init='k-means++', n_init=1)
            kmeans = KMeans(n_clusters=clusters, init='random', n_init=10)
            kmeans.fit_predict(featureMatrix)
            labels = kmeans.labels_
            # print(kmeans.inertia_)
            # print(kmeans.labels_)
            collections = [[] for i in range (clusters)]
            for i in range (len(boundingBoxes)):
                x1 = boundingBoxes[i][0]
                y1 = boundingBoxes[i][1]
                x2 = boundingBoxes[i][0] + boundingBoxes[i][2]
                y2 = boundingBoxes[i][1] + boundingBoxes[i][3]
                croppedImage = frameCopy[y1:y2, x1:x2]
                collections[labels[i]].append(croppedImage.copy())
                cv.putText(frame, str(labels[i]), (boundingBoxes[i][0]+20, boundingBoxes[i][1]), cv.FONT_HERSHEY_SIMPLEX, 0.5 , (0,255,255))
            
            collectionIndex = int(len(collections)/2)
            # newHistograms = [[] for i in range (clusters)]
            newHistograms = [None]*clusters
            for i in range (clusters):
                newHistograms[i] = cbtf(collections[collectionIndex], collections[i], bins)

            featureMatrix = np.zeros((len(boundingBoxes),(3*bins)), dtype='float32')
            counters = [0]*clusters
            for i,label in enumerate(labels):
                features = np.asarray([])
                for j,col in enumerate(colour):
                    hist = newHistograms[label][counters[label]][j]
                    features = np.concatenate((features, hist.T[0]))
                featureMatrix[i] = features
                counters[label] += 1
            
            clusters = 2
            kmeans = KMeans(n_clusters=clusters, init='random', n_init=10)
            kmeans.fit_predict(featureMatrix)
            labels = kmeans.labels_
            # print(labels)
            for i in range (len(boundingBoxes)):
                cv.putText(frame, str(labels[i]), (boundingBoxes[i][0]+40, boundingBoxes[i][1]), cv.FONT_HERSHEY_SIMPLEX, 0.5 , (255,0,255))

        # clusters = 2
        # if (len(boundingBoxes) >= clusters) and (frameNumber>=90):

        #     # baseImageIndex = int(len(boundingBoxes)/2)
        #     # x1 = boundingBoxes[baseImageIndex][0]
        #     # y1 = boundingBoxes[baseImageIndex][1]
        #     # x2 = boundingBoxes[baseImageIndex][0] + boundingBoxes[baseImageIndex][2]
        #     # y2 = boundingBoxes[baseImageIndex][1] + boundingBoxes[baseImageIndex][3]
        #     # baseImage = frameCopy[y1:y2, x1:x2]

        #     cumHists = cumulativeBrightness(croppedImages, bins)

        #     for i in range (len(boundingBoxes)):
        #         x1 = boundingBoxes[i][0]
        #         y1 = boundingBoxes[i][1]
        #         x2 = boundingBoxes[i][0] + boundingBoxes[i][2]
        #         y2 = boundingBoxes[i][1] + boundingBoxes[i][3]
        #         croppedImage = frameCopy[y1:y2, x1:x2]

        #         # if (i==0):
        #         #     baseImage = croppedImage

        #         # histograms = btf(baseImage, croppedImage, bins)
        #         histograms = btfHybrid(cumHists, croppedImage, bins)

        #         features = np.asarray([])
        #         for j in range(3):
        #             features = np.concatenate((features, histograms[j].T[0]))
        #         featureMatrix[i] = features

        #     kmeans = KMeans(n_clusters=clusters, init='random', n_init=10)
        #     kmeans.fit_predict(featureMatrix)
        #     labels = kmeans.labels_
        #     print(labels)
        #     for i in range (len(boundingBoxes)):
        #         cv.putText(frame, str(labels[i]), (boundingBoxes[i][0]+40, boundingBoxes[i][1]), cv.FONT_HERSHEY_SIMPLEX, 0.5 , (255,0,255))

        # Current frame counter
        cv.rectangle(frame, (10, 2), (100,20), (255,255,255), -1)
        cv.putText(frame, str(frameNumber), (15, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))

        # Display the resulting frame
        cv.imshow('Frame', frame)
        cv.imshow('Minimap', minimap)

        # # Pause at frames
        # if (frameNumber == 40 or frameNumber == 50 or frameNumber == 60 or frameNumber == 70 or frameNumber == 80):
        #     cv.waitKey(0)

        # # Pause at frames
        # if (frameNumber == 65 or frameNumber == 66 or frameNumber == 67 or frameNumber == 68 or frameNumber == 69):
        #     cv.waitKey(0)

        if (frameNumber>=startingFrame):
            print("READY!")
            cv.waitKey(0)

        # Pause at frames
        if (frameNumber == 90 or frameNumber == 270 or frameNumber == 450 or frameNumber == 720 or frameNumber == 900):

            labImage = cv.cvtColor(frame, cv.COLOR_BGR2LAB)
            # cv.imshow('Lab', labImage)
            # cv.setMouseCallback('Lab', labValues)

            # print(histogram)
            # print("------")
            # print(histogram.shape)
            # print("------")
            # print(histogram.T[0])
            # print("------")
            # print(histogram.T[0].shape)
            # print(features)
            # print("------")
            # print(features[15][2])
            # print(features.shape)

            # kmeans = KMeans(n_clusters=3)
            # kmeans.fit_predict(featureMatrix)
            # print(kmeans.inertia_)
            # print(kmeans.labels_)
            # print(len(kmeans.labels_))
            # print(len(boundingBoxes))

            # # RGB Histograms
            # colour = ('b','g','r')
            # fig, ax = plt.subplots(len(players), 2)
            # plt.subplots_adjust(left=0.2,
            #         bottom=0.1,
            #         right=0.9,
            #         top=0.9,
            #         wspace=0.2,
            #         hspace=0.6)
            # bins = 64
            
            # for i in range (len(players)):
            #     x1 = boundingBoxes[players[i]][0]
            #     y1 = boundingBoxes[players[i]][1]
            #     x2 = boundingBoxes[players[i]][0] + boundingBoxes[players[i]][2]
            #     y2 = boundingBoxes[players[i]][1] + boundingBoxes[players[i]][3]

            #     croppedImage = frameCopy[y1:y2, x1:x2]

            #     # Create SIFT feature extractor
            #     sift = cv.xfeatures2d.SIFT_create()

            #     # Detect features from the image
            #     keypoints, descriptors = sift.detectAndCompute(croppedImage, None)

            #     # Draw the detected key points
            #     sift_image = cv.drawKeypoints(croppedImage, keypoints, croppedImage)

            #     cv.imshow('PlayerMask '+str(players[i]), sift_image)
                
            #     ax[i][0].set_title("Player "+str(players[i])+" Image RGB Histogram - 256 bin")
            #     ax[i][0].set_xlabel("Bins")
            #     ax[i][0].set_ylabel("# of Pixels")
            #     ax[i][0].set_xlim([0,256])
            #     # ax[i][0].set_ylim([0,0.1])

            #     ax[i][1].set_title("Player "+str(players[i])+" Image RGB Histogram - "+str(bins)+" bin")
            #     ax[i][1].set_xlabel("Bins")
            #     ax[i][1].set_ylabel("# of Pixels")
            #     ax[i][1].set_xlim([0,bins-1])
            #     # ax[i][1].set_ylim([0,0.5])

            #     for j,col in enumerate(colour):
            #         histr = cv.calcHist([croppedImage],[j],None,[256],[0,256])
            #         # histr /= histr.sum()
            #         histr -= histr.mean()
            #         histr /= histr.std()
            #         ax[i][0].plot(histr, color=col)

            #         histr2 = cv.calcHist([croppedImage],[j],None,[bins],[0,256])
            #         # histr2 /= histr2.sum()
            #         histr2 -= histr2.mean()
            #         histr2 /= histr2.std()
            #         ax[i][1].plot(histr2, color=col)

            #         print(histr.mean())
            #         print(histr.std())
            #         print(histr2.mean())
            #         print(histr2.std())
            
            # plt.show()
            
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