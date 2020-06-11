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
    hand_hist = cv.calcHist([roi], [0, 1], None, [180, 256], [0, 180, 0, 256])
    return cv.normalize(hand_hist, hand_hist, 0, 255, cv.NORM_MINMAX)


def hist_masking(frame, hist):
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    dst = cv.calcBackProject([hsv], [0, 1], hist, [0, 200, 0, 256], 1)

    disc = cv.getStructuringElement(cv.MORPH_ELLIPSE, (31, 31))
    cv.filter2D(dst, -1, disc, dst)
    ret, thresh = cv.threshold(dst, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
    # thresh = cv.adaptiveThreshold(dst, 255, cv.ADAPTIVE_THRESH_MEAN_C,
    # 							  cv.THRESH_BINARY, 11, 2)

    thresh = cv.merge((thresh, thresh, thresh))
    cv.imshow("thresh w/o morph", thresh)

    # Close Morphology:

    kernel = np.ones((33, 25), np.uint8)
    thresh = cv.dilate(thresh, kernel, iterations = 1)
    kernel = np.ones((21, 21), np.uint8)
    thresh = cv.erode(thresh, kernel, iterations = 1)

    # kernel = np.ones((7, 7), np.uint8)
    # thresh = cv.erode(thresh, kernel, iterations = 1)
    # kernel = np.ones((25, 25), np.uint8)
    # thresh = cv.dilate(thresh, kernel, iterations = 2)


    # kernel = np.ones((79, 79), np.uint8)
    # thresh = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel)

    cv.imshow("thresh w/ morph", thresh)
    # cv.waitKey(0)

    return cv.bitwise_and(frame, thresh)


def manage_image_opr(frame, hand_hist):
    hist_mask_image = hist_masking(frame, hand_hist)

    hist_mask_image = cv.erode(hist_mask_image, None, iterations=2)
    hist_mask_image = cv.dilate(hist_mask_image, None, iterations=2)

    contour_list = contours(hist_mask_image)
    max_cont = max(contour_list, key=cv.contourArea)

    cnt_centroid = centroid(max_cont)
    cv.circle(frame, cnt_centroid, 5, [255, 0, 255], -1)

    if max_cont is not None:
        hull = cv.convexHull(max_cont, returnPoints=False)
        defects = cv.convexityDefects(max_cont, hull)
        far_point = farthest_point(defects, max_cont, cnt_centroid)
        print("Centroid : " + str(cnt_centroid) + ", farthest Point : " + str(far_point))
        cv.circle(frame, far_point, 5, [0, 0, 255], -1)
        if len(traverse_point) < 20:
            traverse_point.append(far_point)
        else:
            traverse_point.pop(0)
            traverse_point.append(far_point)

        draw_circles(frame, traverse_point)


# helpers for manage_image_opr-----------------------------------
def draw_circles(frame, traverse_point):
    if traverse_point is not None:
        for i in range(len(traverse_point)):
            cv.circle(frame, traverse_point[i], int(5 - (5 * i * 3) / 100), [0, 255, 255], -1)


def farthest_point(defects, contour, centroid):
    if defects is not None and centroid is not None:
        s = defects[:, 0][:, 0]
        cx, cy = centroid

        x = np.array(contour[s][:, 0][:, 0], dtype=np.float)
        y = np.array(contour[s][:, 0][:, 1], dtype=np.float)

        xp = cv.pow(cv.subtract(x, cx), 2)
        yp = cv.pow(cv.subtract(y, cy), 2)
        dist = cv.sqrt(cv.add(xp, yp))

        dist_max_i = np.argmax(dist)

        if dist_max_i < len(s):
            farthest_defect = s[dist_max_i]
            farthest_point = tuple(contour[farthest_defect][0])
            return farthest_point
        else:
            return None


def centroid(max_contour):
    moment = cv.moments(max_contour)
    if moment['m00'] != 0:
        cx = int(moment['m10'] / moment['m00'])
        cy = int(moment['m01'] / moment['m00'])
        return cx, cy
    else:
        return None


def contours(hist_mask_image):
    gray_hist_mask_image = cv.cvtColor(hist_mask_image, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(gray_hist_mask_image, 0, 255, 0)
    _, cont, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    return cont


if __name__ == '__main__':
    # Testing using images before a video
    img = cv.imread("christianshand2.jpg")

    if img is None:
        sys.exit("Could not read the image")

    cv.imshow("original image", img)
    cv.waitKey(0)

    img_rect = draw_rect(img)
    cv.imshow("img with drawn rectangles", img_rect)
    cv.waitKey(0)

    hist = hand_histogram(img)
    img_mask = hist_masking(img, hist)

    cv.imshow("masked img", img_mask)
    cv.waitKey(0)

    # Testing with Recorded Video:
    vid = cv.VideoCapture("christian1.mov")

    while vid.isOpened():
        ret, frame = vid.read()

        # Check frame is read correctly
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        w = int(frame.shape[1] * 50 / 100)
        h = int(frame.shape[0] * 50 / 100)
        frame = cv.resize(frame, (w, h))
        frame = cv.rotate(frame, cv.ROTATE_180)
        # manage_image_opr(frame, hist)
        # cv.imshow("actual vid",frame)

        # 		# while not calibrate:
        # 		frame_rect = draw_rect(frame)
        # 		rows, cols, _ = frame.shape
        #
        # 		font = cv.FONT_HERSHEY_PLAIN
        # 		cv.putText(frame, "Press 'k' key while holding hand on green grid",
        # 				   (int(1/10*rows), int(2/3*cols)), font, 1.2, (0, 255, 100), 2, cv.LINE_AA)
        # 		cv.putText(frame, "for calibration",
        # 				   (int(1/10*rows), int(2/3*cols+25)), font, 1.2, (0, 255, 100), 2, cv.LINE_AA)
        #
        # 		cv.imshow('video', frame_rect)
        #
        # if cv.waitKey(1) == ord('k'):
        # hist = hand_histogram(frame)

        # used for testing
        img_mask = hist_masking(frame, hist)
        cv.imshow("masked vid", img_mask)
        # cv.waitKey(0)


        if cv.waitKey(10) == ord('q'):
            break

    vid.release()
    cv.destroyAllWindows()

# --------------------------------------------------------------------
# # Testing with live video
# 	calibrate = False
# 	cap = cv.VideoCapture(0)
# 	if not cap.isOpened():
# 		print("Can't open camera")
# 		exit()
#
# 	while not calibrate:
# 		# Capture frame-by-frame
# 		ret, frame = cap.read()
# 		if not ret:
# 			print("can't receive frame(stream end?). Exiting...")
# 			break
#
# 		# Video Operations
# 		frame = cv.flip(frame, 1)
#
# 		# while not calibrate:
# 		frame_rect = draw_rect(frame)
# 		rows, cols, _ = frame.shape
#
# 		font = cv.FONT_HERSHEY_PLAIN
# 		cv.putText(frame, "Press 'k' key while holding hand on green grid",
# 				   (int(1/10*rows), int(2/3*cols)), font, 1.2, (0, 255, 100), 2, cv.LINE_AA)
# 		cv.putText(frame, "for calibration",
# 				   (int(1/10*rows), int(2/3*cols+25)), font, 1.2, (0, 255, 100), 2, cv.LINE_AA)
#
# 		cv.imshow('video capture', frame_rect)
#
# 		if cv.waitKey(1) == ord('k'):
# 			hist = hand_histogram(frame)
# 			img_mask = hist_masking(frame, hist)
# 			cv.imshow("masked vid", img_mask)
# 			cv.waitKey(0)
#
# 		if cv.waitKey(1) == ord('q'):
# 			break
#
# 	while True:
# 		# Capture frame-by-frame
# 		ret, frame = cap.read()
# 		if not ret:
# 			print("can't receive frame(stream end?). Exiting...")
# 			break
#
# 		cv.imshow('video capture', frame)
#
#
# 		# Video Operations
# 		if cv.waitKey(1) == ord('q'):
# 			break
#
# 	cap.release()
# 	cv.destroyAllWindows()
