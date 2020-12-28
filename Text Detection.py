from imutils.object_detection import non_max_suppression
import numpy as np
import argparse
import time
import cv2

# load the input image and grab the image dimensions
image = cv2.imread('/Users/aryan.sharma/Downloads/Visum1.jpg')
orig = image.copy()
(H, W) = image.shape[:2]

# set the new width and height and then determine the ratio in change
# for both the width and height
(newW, newH) = (320, 320)
rW = W/ float(newW)
rH = H/ float(newH)

#resize the image and grab the new image dimensions
image = cv2.resize(image, (newW, newH))
(H, W) = image.shape[:2]

# define the two output layer names for the EAST detector model that
# we are interested-- the first is the output probabilities and the
# second can be used to derive the bounding box coordinates of text

layerNames = [
    "feature_fusion/Conv_7/Sigmoid",
    "feature_fusion/concat_3"]

# load the pre-trained EAST text detector
print("[INFO] loading EAST text detector...")
net = cv2.dnn.readNet(["east"])

# construct a blob from the image and then perform a forward pass of
# the model to obbtain the two output layer sets
blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
                             (123.68, 116.78, 103.94), swapRB=True, crop=False)
start = time.time()
net.setInput(blob)
(scores, geometry) = net.forward(layerNames)
end = time.time()

# show timing information on text prediction
print("[INFO] TEXT DETECTION TOOK {: 6F} seconds".format(end - start))

# grab the number of rows and columns from the scores volume, then
# initialise our set of bounding box rectangles and corresponding
# confidence scores
(numRows, numCols) = scores.shape[2:4]
rects = []
confidences = []

# loop over the number of rows
for y in range(0, numRows):
    # extract the scores (probbilities), followed by the geometrical
    # data used to derive potential bounding box coordinates that
    # surround text
    scoresData = scores[0, 0, y]
    xData0 = geometry[0, 0, y]
    xData1 = geometry[0, 1, y]
    xData2 = geomtery[0, 2, y]
    xData3 = geomtery[0, 3, y]
    anglesData = geometry[0, 4, y]

    # loop over number of columns
    for x in range(0, numCols):
        # if our score does not have sufficient probability, ignore it
        if scoresData[x] < args["min_confidence"]:
            continue

        # compute the offset factor as our resulting feature maps will
        # be 4x smaller than the input image
        (offsetX, offsetY) = (x * 4.0, y * 4.0)

        # extract the rotation angle for the prediciton and then
        # compute the sin and cosine
        angle = anglesData[x]
        cos = np.cos(angle)
        sin = np.sin(angle)

        # use the geometry volume to derive width and height of bounding box
        h = xData0[x] + xData2[x]
        w = xData1[x] + xData3[x]

        # compute both the starting and ending (x, y) coordinates for the text bounding box
        endX = int(offsetX + (cos * xData[x]) + (sin * xData2[x]))
        endY = int(offsetY - (sin * xData[x]) + (cos * xData[x]))
        startX = int(endX - w)
        startY = int(endY - h)

        # add the bounding box coordinates and probability score to our respective lists
        rects.append((startX, startY, endX, endY))
        confidences.append(scoresData[x])

# apply the non-maxima suppression to suppress weka, overlapping bounding boxes
boxes = non_max_suppression(np.array(rects), probs=confidences)

# loop over the bounding boxes
for (startX, startY, endX, endY) in boxes:
    # scale the bounding box coordinates based on respective ratios
    startX = int(startX * rW)
    startY = int(startY * rH)
    endX = int(endX * rW)
    endY = int(endY * rH)

    # draw the bounding bo on the image
    cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 2)

# show the output image
cv2.imshow("Text Detection", orig)
cv2.waitkey(0)


