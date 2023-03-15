# Library imports
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import os

def cumulativeBrightness(c):
    # Initialisation
    cumulativeHists = np.zeros((3,256,1), dtype='float32')

    # Amalgamation of image histograms
    for image in c:
        for i in range(3):
            cumulativeHists[i] += cv.calcHist([image],[i],None,[256],[0,256])

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
    
    return newHists


folderDir = "C:/Users/jeeva/OneDrive/Documents/Course/Year 3/COMP30040/Datasets/Aphrodite Dropbox/2players"
folderContents = os.listdir(folderDir)

colour = ('b','g','r')
bins = 256

featureMatrix = np.zeros((len(folderContents),(3*bins)), dtype='float32')

clusters = 2

defaultImage1 = cv.imread("Datasets/Aphrodite Dropbox/2players/playerA_01122.jpg")
defaultImage2 = cv.imread("Datasets/Aphrodite Dropbox/2players/playerB_12001.jpg")

for i in range(len(folderContents)):
    print("Currently on:",i)
    # print(folderContents[i])
    image = cv.imread("Datasets/Aphrodite Dropbox/2players/"+folderContents[i])
    # hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    # lab = cv.cvtColor(image, cv.COLOR_BGR2Lab)
    # cv.imshow('Image', image)
    # cv.waitKey(0)

    # cv.imshow('Player'+str(i), lab)

    # ab = lab.copy().T
    # ab[0] = np.zeros(ab[0].shape, dtype='uint8')
    # ab = ab.T

    # cv.imshow('AB Player'+str(i), ab)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    newHists = cbtf([defaultImage1, defaultImage2], [image])[0]

    features = np.asarray([])
    for j,col in enumerate(colour):
        # histogram = cv.calcHist([lab],[j],None,[bins],[0,256])
        # histogram = cv.calcHist([image],[j],None,[bins],[0,256])
        histogram = newHists[j]
        histogram /= histogram.sum()
        # histogram -= histogram.mean()
        # histogram /= histogram.std()
        features = np.concatenate((features, histogram.T[0]))
    featureMatrix[i] = features

kmeans = KMeans(n_clusters=clusters, init='random', n_init=10)
labels = kmeans.fit_predict(featureMatrix)
print(kmeans.inertia_)
print(kmeans.labels_)

print("Done!")

playerA = [0]*clusters
playerB = [0]*clusters
for i in range(len(folderContents)):

    # # Console output
    # print(folderContents[i])
    # print(labels[i])
    # print("--------")

    # # Display image
    # image = cv.imread("Datasets/Aphrodite Dropbox/2players/"+folderContents[i])
    # cv.imshow('Image', image)
    # cv.waitKey(0)

    # Measurements
    if folderContents[i].startswith("playerA"):
        playerA[labels[i]] += 1
    else:
        playerB[labels[i]] += 1

print("Final Results:")
print(playerA)
print(playerB)