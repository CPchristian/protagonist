# CPE428 Computer Vision Final Project Notes
##### Author Jenny Chiao
---
###Contents:

---
## Skin Color Detection
Idea is that the person holds their hand up to where some screen drawn 
rectangles are for the program to pick up skin color samples.

The samples are then used to generate an HSV histogram to be used for 
the rest of the program. 

> source: https://dev.to/amarlearning/finger-detection-and-tracking-using-opencv-and-python-586m

* `img.shape` returns tple of number of rows, columns, and channels (if img is color)
* `cv.cvtColor()` used to transform frame from BGR to HSV (hue saturation values)
* `cv.calcHist()` creates histogram using region of interest matrix (the green drawn rectangles)
* `cv.normalize()` normalizes matrix
* `cv.calcBackProject` **Back Projection**
    * uses histogram to separate features in an img
    * used to apply skin color histogram to a frame
    * Histogram Back Projection - image segmentation or finding objects of interest in an image. It creates
    an image of the same size but single channel as the input image. Each pixel corresponds to the probability
     of the pixel belonging to an object. The output img will be black and white. The white is the matching 
     color sample.
    > https://docs.opencv.org/master/dc/df6/tutorial_py_histogram_backprojection.html>
    > https://docs.opencv.org/2.4/doc/tutorials/imgproc/histograms/back_projection/back_projection.html
    
    * range parameter seems to adjust tolerance?


* `cv.filter2D` and `cv.threshold` used to smoothen the image
* `cv.bitwise_and` used to mask input frame 
     dfd