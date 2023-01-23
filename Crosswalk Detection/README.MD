## Crosswalk detection using various OpenCV operations

This basic algorithm can detect crosswalks based on the geometric property that the lines are invariant to movement and the lines remain parallel.
Moreover, each component of a crosswalk represents a white rectangle (basic geometric object with 4 edges). Since the resulting contour approximations are not perfect we have to take into consideration possible approximations of polygons with more edges. 

## Examples:

![dataset/crosswalk5](/assets/images/tux.png) 

![results/result3](/assets/images/tux.png) 

## Improvements

- More defined preprocessing steps according to the capturing device.
- Better binarization process for the image.
- Consider other possible morphological operations.

**NOTE**: Project might require C++ version 17 or above.