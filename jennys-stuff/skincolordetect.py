# skincolordetect.py
'''User holds hand up to screen where rectangles are to calibrate for skin colors.'''

import sys
import cv2 as cv
import numpy as np

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

	# cv.rectangle draws each of the  9 green rectangles on the frame.
	for i in range(tot_rect):
		cv.rectangle(frame, (hand_rect_y1[i], hand_rect_x1[i]),
					 (hand_rect_y2[i], hand_rect_x2[i]),
					 (0, 255, 0), 1)

	return frame

# gets skin color samples from rectangles and converts to histogram
def hand_histogram(frame):
	global hand_rect_x1, hand_rect_y1

	hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
	# region of interest
	roi = np.zeros([90, 10, 3], dtype=hsv_frame.dtype)

	for i in range(tot_rect):
		roi[i * 10: i * 10 + 10, 0:10] = hsv_frame[hand_rect_x1[i]: hand_rect_x1[i] + 10,
										 hand_rect_y1[i]: hand_rect_y1[i] + 10]

	hand_hist = cv.calcHist([roi], [0, 1], None, [180, 256],[0, 180, 0, 256])
	return cv.normalize(hand_hist, hand_hist, 0, 255, cv.NORM_MINMAX)


def hist_masking(frame, hist):
	hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
	dst = cv.calcBackProject([hsv], [0,1], hist, [0, 180, 0, 256], 1)

	disc = cv.getStructuringElement(cv.MORPH_ELLIPSE, (31,31))
	cv.filter2D(dst, -1, disc, dst)

	ret, thresh = cv.threshold(dst, 150, 255, cv.THRESH_BINARY)

	thresh = cv.merge((thresh, thresh, thresh))

	return cv.bitwise_and(frame, thresh)



# Testing using images before a video
# img = cv.imread("test.jpg")
#
# if img is None:
# 	sys.exit("Could not read the image")
#
# cv.imshow("original image", img)
# cv.waitKey(0)
#
# img_rect = draw_rect(img)
# cv.imshow("img with drawn rectangles", img_rect)
# cv.waitKey(0)


# Testing with live video
cap = cv.VideoCapture(0)

if not cap.isOpened():
	print("Can't open camera")
	exit()

while True:
	# Capture frame-by-frame
	ret, frame = cap.read()

	if not ret:
		print("can't receive frame(stream end?). Exiting...")
		break

	# Video Operations
	frame = cv.flip(frame, 1)
	frame_rect = draw_rect(frame)

	cv.imshow('video capture', frame_rect)
	if cv.waitKey(1) == ord('q'):
		break
cap.release()
cv.destroyAllWindows()
