import cv2
import numpy as np
from collections import deque
import sys

hand_hist = None
traverse_point = []
tot_rect = 9
hand_rect_x1 = None
hand_rect_y1 = None
hand_rect_x2 = None
hand_rect_y2 = None


def draw_rect(frame):
	rows, cols, _ = frame.shape
	global tot_rect, hand_rect_x1, hand_rect_x2, hand_rect_y1, hand_rect_y2

	hand_rect_x1 = np.array(
		[6 * rows / 20, 6 * rows / 20, 6 * rows / 20, 9 * rows / 20, 9 * rows / 20, 9 * rows / 20, 12 * rows / 20,
		 12 * rows / 20, 12 * rows / 20], dtype=np.uint32)

	hand_rect_y1 = np.array(
		[9 * cols / 20, 10 * cols / 20, 11 * cols / 20, 9 * cols / 20, 10 * cols / 20, 11 * cols / 20, 9 * cols / 20,
		 10 * cols / 20, 11 * cols / 20], dtype=np.uint32)

	hand_rect_x2 = hand_rect_x1 + 10
	hand_rect_y2 = hand_rect_y1 + 10

	# cv2.rectangle draws each of the  9 green rectangles on the frame.
	for i in range(tot_rect):
		cv2.rectangle(frame, (hand_rect_y1[i], hand_rect_x1[i]),
					  (hand_rect_y2[i], hand_rect_x2[i]),
					  (0, 255, 0), 1)

	return frame


# gets skin color samples from rectangles and converts to histogram
def hand_histogram(frame):
	global hand_rect_x1, hand_rect_y1

	hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
	cv2.imshow("HSV", hsv_frame)
	cv2.waitKey(0)
	# region of interest
	roi = np.zeros([90, 10, 3], dtype=hsv_frame.dtype)

	for i in range(tot_rect):
		roi[i * 10: i * 10 + 10, 0:10] = hsv_frame[hand_rect_x1[i]: hand_rect_x1[i] + 10,
										 hand_rect_y1[i]: hand_rect_y1[i] + 10]
	hand_hist = cv2.calcHist([roi], [0, 1], None, [180, 256], [0, 180, 0, 256])
	return cv2.normalize(hand_hist, hand_hist, 0, 255, cv2.NORM_MINMAX)


def hist_masking(frame, hist):
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
	dst = cv2.calcBackProject([hsv], [0, 1], hist, [0, 180, 0, 256], 1)  # changed

	disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
	cv2.filter2D(dst, -1, disc, dst)
	ret, thresh = cv2.threshold(dst, 0, 255, cv2.THRESH_BINARY)
	# ret, thresh = cv2.threshold(dst, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	# thresh = cv2.adaptiveThreshold(dst, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
	# 							  cv2.THRESH_BINARY, 11, 2)

	thresh = cv2.merge((thresh, thresh, thresh))
	cv2.imshow("thresh w/o morph", thresh)

	# Close Morphology:

	kernel = np.ones((19, 19), np.uint8)
	thresh = cv2.dilate(thresh, kernel, iterations=1)
	kernel = np.ones((7, 7), np.uint8)
	thresh = cv2.erode(thresh, kernel, iterations=1)

	# kernel = np.ones((79, 79), np.uint8)
	# thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

	cv2.imshow("thresh w/ morph", thresh)
	# cv2.waitKey(0)

	# return cv2.bitwise_and(frame, thresh)

	thresh = thresh.astype(np.uint8)
	thresh = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
	return thresh


# ---------------------------------------------------
# Color Sampling from Img:
img = cv2.imread("bluecap.jpg")

if img is None:
	sys.exit("Could not read the image")

cv2.imshow("original image", img)
cv2.waitKey(0)

img_rect = draw_rect(img)
cv2.imshow("img with drawn rectangles", img_rect)
cv2.waitKey(0)

hist = hand_histogram(img)
img_mask = hist_masking(img, hist)

cv2.imshow("masked img", img_mask)
cv2.waitKey(0)

# ---------------------------------------------------
# Christian's code:


# load in the video of red circle moved around
cap = cv2.VideoCapture('bluecapvid.mov')

# define the upper and lower boundaries for a color to be considered "blue"
# (blue, green, red)
red_lower = np.array([110, 60, 60])
red_upper = np.array([140, 255, 255])

# define a 5x5 kernel for erosion and dilation
kernel = np.ones((5, 5), np.uint8)
# g = cv2.getGaussianKernel(31, 5)

# initialize deques to store different colors in different arrays
bpoints = [deque(maxlen=512)]
gpoints = [deque(maxlen=512)]
rpoints = [deque(maxlen=512)]
ypoints = [deque(maxlen=512)]

# initialize an index variable for each of the colors
bindex = 0
gindex = 0
rindex = 0
yindex = 0

# just a handy array and an index variable to get the color of interest on the go
# Blue, Green, Red, Yellow respectively
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
colorIndex = 3

# Create a blank white image
paintWindow = np.zeros((720, 1280, 3)) + 255

# setup the paint interface
paintWindow = cv2.rectangle(paintWindow, (40, 1), (240, 65), (0, 0, 0), 2)
paintWindow = cv2.rectangle(paintWindow, (260, 1), (455, 65), colors[0], -1)
paintWindow = cv2.rectangle(paintWindow, (475, 1), (670, 65), colors[1], -1)
paintWindow = cv2.rectangle(paintWindow, (690, 1), (885, 65), colors[2], -1)
paintWindow = cv2.rectangle(paintWindow, (905, 1), (1100, 65), colors[3], -1)
cv2.putText(paintWindow, "CLEAR ALL", (90, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "BLUE", (336, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "GREEN", (540, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "RED", (770, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "YELLOW", (970, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 2, cv2.LINE_AA)

# Create a window to display the above image (later)]
cv2.namedWindow('Paint', cv2.WINDOW_AUTOSIZE)

# for live webcam processing
# camera = cv2.VideoCapture(0)
# #
# if not camera.isOpened():
#     print("cant open camera")
# #     exit()

# traverse through each video
while cap.isOpened():

	""" for live webcam processing (video resolution TBD) """
	# (ret, frame) = camera.read()
	# frame = cv2.flip(frame, 1)
	# hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

	""" for recorded video processing (video resolution in iPhone XsMax: (720, 1280) @ 30 fps """
	ret, frame = cap.read()
	# fip the frame so its upright=
	frame = cv2.flip(frame, 1)
	# hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

	# check if at the end of the video and break out of while loop
	if not ret:
		print('\n Done showing video. Exiting. . .')
		break

	# place colored rectangles at the top of each frame
	frame = cv2.rectangle(frame, (40, 1), (240, 65), (122, 122, 122), -1)  # clear all
	frame = cv2.rectangle(frame, (260, 1), (455, 65), colors[0], -1)  # blue
	frame = cv2.rectangle(frame, (475, 1), (670, 65), colors[1], -1)  # green
	frame = cv2.rectangle(frame, (690, 1), (885, 65), colors[2], -1)  # red
	frame = cv2.rectangle(frame, (905, 1), (1100, 65), colors[3], -1)  # yellow
	cv2.putText(frame, "CLEAR ALL", (90, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
	cv2.putText(frame, "BLUE", (336, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
	cv2.putText(frame, "GREEN", (540, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
	cv2.putText(frame, "RED", (770, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
	cv2.putText(frame, "YELLOW", (970, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 2, cv2.LINE_AA)

	# # converts BGR -> HSV
	# hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #
	# # find the contour of interest (the blue cap)
	# redMask = cv2.inRange(hsv, red_lower, red_upper)
	# redMask = cv2.erode(redMask, kernel, iterations=2)
	# redMask = cv2.morphologyEx(redMask, cv2.MORPH_OPEN, kernel)
	# redMask = cv2.dilate(redMask, kernel, iterations=1)
	# cv2.imshow("redmask", redMask)
	# cv2.waitKey(0)
	# print("redMask.copy type is : ", type(redMask.copy()))

	frame_mask = hist_masking(frame, hist)
	# print("frame_mask: ", type(frame_mask.type()))

	# find contours in the image
	# cnts, _  = cv2.findContours(redMask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	_, cnts, _ = cv2.findContours(frame_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cv2.drawContours(frame, cnts, -1, (255, 255, 0))

	# the if condition passes when a contour is found
	# check if any contour (red stuff) is were found (it'll be fingertips here later)
	if len(cnts) > 0:
		# sort the contours and find the largest one -- we assume this contour corresponds to the area of the cap
		cnt = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
		# get the radius of the enclosing circle around the found contour
		((x, y), radius) = cv2.minEnclosingCircle(cnt)
		# draw the circle around the contour
		cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
		# get the moments to calculate the center of the contour (in this case a circle)
		M = cv2.moments(cnt)
		center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))

		# center is above the 65th pixel in the vertical direction
		if center[1] <= 65:
			# check the center's position horizontally
			if 40 <= center[0] <= 240:  # Clear All
				bpoints = [deque(maxlen=512)]
				gpoints = [deque(maxlen=512)]
				rpoints = [deque(maxlen=512)]
				ypoints = [deque(maxlen=512)]

				bindex = 0
				gindex = 0
				rindex = 0
				yindex = 0

				paintWindow[67:, :, :] = 255

			elif 260 <= center[0] <= 455:
				colorIndex = 0  # Blue
			elif 475 <= center[0] <= 670:
				colorIndex = 1  # Green
			elif 690 <= center[0] <= 885:
				colorIndex = 2  # Red
			elif 905 <= center[0] <= 1100:
				colorIndex = 3  # Yellow
		else:
			if colorIndex == 0:
				bpoints[bindex].appendleft(center)
			elif colorIndex == 1:
				gpoints[gindex].appendleft(center)
			elif colorIndex == 2:
				rpoints[rindex].appendleft(center)
			elif colorIndex == 3:
				ypoints[yindex].appendleft(center)

	# Append the next deque when no contours are detected (i.e., bottle cap not visible)
	else:
		bpoints.append(deque(maxlen=512))
		bindex += 1
		gpoints.append(deque(maxlen=512))
		gindex += 1
		rpoints.append(deque(maxlen=512))
		rindex += 1
		ypoints.append(deque(maxlen=512))
		yindex += 1

	# Draw lines of all the colors (Blue, Green, Red and Yellow)
	points = [bpoints, gpoints, rpoints, ypoints]

	for i in range(len(points)):
		for j in range(len(points[i])):
			for k in range(1, len(points[i][j])):
				if points[i][j][k - 1] is None or points[i][j][k] is None:
					continue
				cv2.line(frame, points[i][j][k - 1], points[i][j][k], colors[i], 2)
				cv2.line(paintWindow, points[i][j][k - 1], points[i][j][k], colors[i], 2)

	# Show the frame and the paintWindow image
	cv2.imshow("Tracking", frame)
	cv2.imshow("Paint", paintWindow)

	# If the 'q' key is pressed, stop the loop
	if cv2.waitKey(33) & 0xFF == ord("q"):
		break

# for live webcam processing
# camera.release()
cv2.destroyAllWindows()
