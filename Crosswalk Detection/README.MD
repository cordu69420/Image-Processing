## Crosswalk detection using various OpenCV operations

This basic algorithm can detect crosswalks based on the geometric property that the lines are invariant to movement and remain parallel.
Moreover, each component of a crosswalk represents a white rectangle (basic geometric object with 4 edges). Since the resulting contour approximations are not perfect we have to take into consideration possible approximations of polygons with more edges. 

## Examples:

![Crosswalk example](https://raw.githubusercontent.com/cordu69420/Image-Processing/master/Crosswalk%20Detection/dataset/crosswalk5.jpg) 

![Crosswalk example result](https://raw.githubusercontent.com/cordu69420/Image-Processing/master/Crosswalk%20Detection/results/result3.jpg) 

## Improvements

- More defined preprocessing steps according to the capturing device.
- Better binarization process for the image.
- Consider other possible morphological operations.

**NOTE**: Project might require C++ version 17 or above.
