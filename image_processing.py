from imutils.object_detection import non_max_suppression
import numpy as np
import argparse
import time
import cv2
import pytesseract
import imutils

def adjust_gamma(image, gamma=1.0):
    invGamma=1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
    for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)

# load the input image and grab the image dimensions/Users/nitesh.kumar/Downloads/opencv-text-recognition/images/accord-exhibit-andheri-east-mumbai-sign-board-manufacturers-b6a124sccs.jpg
image = cv2.imread('/Users/aryan.sharma/Downloads/Screenshot_20200630-183020_Gallery.jpg')
orig = image.copy()
(H, W) = image.shape[:2]

# set the new width and height and then determine the ratio in change
# for both the width and height
(newW, newH) = (320,640)
rW = W / float(newW)
rH = H / float(newH)

# resize the image and grab the new image dimensions
image = cv2.resize(image, (newW, newH))
(H, W) = image.shape[:2]

# define the two output layer names for the EAST detector model that
# we are interested -- the first is the output probabilities and the
# second can be used to derive the bounding box coordinates of text
layerNames = [
"feature_fusion/Conv_7/Sigmoid",
"feature_fusion/concat_3"]

# load the pre-trained EAST text detector
print("[INFO] loading EAST text detector...")
net = cv2.dnn.readNet('/Users/aryan.sharma/Downloads/opencv-text-detection 3/frozen_east_text_detection.pb')

# construct a blob from the image and then perform a forward pass of
# the model to obtain the two output layer sets
blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
(123.68, 116.78, 103.94), swapRB=True, crop=False)
start = time.time()
net.setInput(blob)
(scores, geometry) = net.forward(layerNames)
end = time.time()

# show timing information on text prediction
print("[INFO] text detection took {:.6f} seconds".format(end - start))

# grab the number of rows and columns from the scores volume, then
# initialize our set of bounding box rectangles and corresponding
# confidence scores
(numRows, numCols) = scores.shape[2:4]
rects = []
confidences = []

# loop over the number of rows
for y in range(0, numRows):
    scoresData = scores[0, 0, y]
    xData0 = geometry[0, 0, y]
    xData1 = geometry[0, 1, y]
    xData2 = geometry[0, 2, y]
    xData3 = geometry[0, 3, y]
    anglesData = geometry[0, 4, y]

# loop over the number of columns
for x in range(0, numCols):
    if scoresData[x] < 0.2:
     continue

# compute the offset factor as our resulting feature maps will
# be 4x smaller than the input image
(offsetX, offsetY) = (x * 4.0, y * 4.0)

# extract the rotation angle for the prediction and then
# compute the sin and cosine
angle = anglesData[x]
cos = np.cos(angle)
sin = np.sin(angle)

# use the geometry volume to derive the width and height of
# the bounding box
h = xData0[x] + xData2[x]
w = xData1[x] + xData3[x]

# compute both the starting and ending (x, y)-coordinates for
# the text prediction bounding box
endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
startX = int(endX - w)
startY = int(endY - h)

# add the bounding box coordinates and probability score to
# our respective lists
rects.append((startX, startY, endX, endY))
confidences.append(scoresData[x])

# apply non-maxima suppression to suppress weak, overlapping bounding
# boxes
boxes = non_max_suppression(np.array(rects), probs=confidences)

# loop over the bounding boxes
for (startX, startY, endX, endY) in boxes:
    startX = int(startX * rW)
    startY = int(startY * rH)
    endX = int(endX * rW)
    endY = int(endY * rH)

# draw the bounding box on the image
cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 2)
orig=orig[startY:endY,startX:endX]

# loop over various values of gamma
for gamma in np.arange(0.0, 1.0, 0.5):
    if gamma == 1:
     continue
# apply gamma correction and show the images
gamma = gamma if gamma > 0 else 0.1
adjusted = adjust_gamma(orig, gamma=gamma)
cv2.putText(adjusted, "g={}".format(gamma), (10, 30),
cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
cv2.imshow("Images", np.hstack([orig, adjusted]))
#print(original)
cv2.waitKey(0)


# threshold the image by setting all pixel values less than 225
# to 255 (white; foreground) and all pixel values >= 225 to 255
# (black; background), thereby segmenting the image
thresh = cv2.threshold(adjusted, 130,255, cv2.THRESH_TOZERO_INV)[1]
#thresh = cv2.threshold(adjusted, 100, 270, cv2.THRESH_TOZERO)[1]
#thresh = cv2.threshold(adjusted, 100, 270, cv2.THRESH_BINARY_INV)[1]
cv2.imwrite('/Users/aryan.sharma/PycharmProjects/untitled2/processed.jpg', thresh)
cv2.waitKey(0)
#cv2.imshow("Thresh", thresh)
#cv2.waitKey(0)