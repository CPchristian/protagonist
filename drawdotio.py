import cv2
import numpy as np
from collections import deque

# load in the video of red circle moved around
cap = cv2.VideoCapture('blue4.mov')

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
paintWindow = cv2.rectangle(paintWindow, (40,1), (240,65), (0,0,0), 2)
paintWindow = cv2.rectangle(paintWindow, (260,1), (455,65), colors[0], -1)
paintWindow = cv2.rectangle(paintWindow, (475,1), (670,65), colors[1], -1)
paintWindow = cv2.rectangle(paintWindow, (690,1), (885,65), colors[2], -1)
paintWindow = cv2.rectangle(paintWindow, (905,1), (1100,65), colors[3], -1)
cv2.putText(paintWindow, "CLEAR ALL", (90, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "BLUE", (336, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "GREEN", (540, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "RED", (770, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "YELLOW", (970, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100,100,100), 2, cv2.LINE_AA)



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

    # converts BGR -> HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # place colored rectangles at the top of each frame
    frame = cv2.rectangle(frame, (40, 1), (240, 65), (122, 122, 122), -1)   # clear all
    frame = cv2.rectangle(frame, (260, 1), (455, 65), colors[0], -1)        # blue
    frame = cv2.rectangle(frame, (475, 1), (670, 65), colors[1], -1)        # green
    frame = cv2.rectangle(frame, (690, 1), (885, 65), colors[2], -1)        # red
    frame = cv2.rectangle(frame, (905, 1), (1100, 65), colors[3], -1)       # yellow
    cv2.putText(frame, "CLEAR ALL", (90, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "BLUE", (336, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "GREEN", (540, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "RED", (770, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "YELLOW", (970, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150,150,150), 2, cv2.LINE_AA)



    # find the contour of interest (the blue cap)
    redMask = cv2.inRange(hsv, red_lower, red_upper)
    redMask = cv2.erode(redMask, kernel, iterations=2)
    redMask = cv2.morphologyEx(redMask, cv2.MORPH_OPEN, kernel)
    redMask = cv2.dilate(redMask, kernel, iterations=1)

    # find contours in the image
    cnts, _  = cv2.findContours(redMask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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










