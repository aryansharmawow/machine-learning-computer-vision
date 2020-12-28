import cv2   # library
import imutils
import numpy as np


def adjust_gamma(image, gamma=1.0):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
	# apply gamma correction using the lookup table
	return cv2.LUT(image, table)

image = cv2.imread("/Users/nitesh.kumar/PycharmProjects/Master_Visum/ravi9.png")
#print(image)
(h, w, d) = image.shape
print("width={}, height={}, depth={}".format(w, h, d))
cv2.imshow("Image", image)
cv2.waitKey(0)
resized = imutils.resize(image, width=1000)
cv2.imshow("Imutils Resize", resized)
cv2.waitKey(0)
median = cv2.medianBlur(resized,2)
resized=median
cv2.imshow("2D Blur", resized)
cv2.waitKey(0)
# loop over various values of gamma
for gamma in np.arange(0, 2, 0.5):
	# ignore when gamma is 1 (there will be no change to the image)
	if gamma == 1:
		continue
	# apply gamma correction and show the images
	gamma = gamma if gamma > 0 else 0.1
	adjusted = adjust_gamma(resized, gamma=gamma)
# convert the image to grayscale
gray = cv2.cvtColor(adjusted, cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray", gray)
cv2.waitKey(0)
thresh = cv2.threshold(gray, 89, 255, cv2.THRESH_OTSU)[1]
cv2.imshow("Thresh", thresh)
cv2.waitKey(0)







