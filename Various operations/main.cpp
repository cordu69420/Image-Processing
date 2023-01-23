#include <opencv2/opencv.hpp>
#include <stdio.h>

using namespace cv;

/**
 * @brief Convert an image to grayscale 8bit.
 *
 * @param input_image The RGB/BGR original image.
 * @return Mat
 */
Mat convertGrayscale(Mat input_image)
{
    Mat output_image = Mat(input_image.size(), CV_8UC1);

    for (int i = 0; i < output_image.rows; i++) {
        for (int j = 0; j < output_image.cols; j++) {
            output_image.at<uchar>(i, j) = (input_image.at<Vec3b>(i, j)[0] + input_image.at<Vec3b>(i, j)[1] + input_image.at<Vec3b>(i, j)[2]) / 3;
        }
    }
    return output_image;
}
/**
 * @brief Strech the constrast of an image such that all the values are used.
 *
 * @param input_image
 * @return Mat
 */
Mat contrastStreching(Mat input_image)
{
    Mat output_image = Mat(input_image.size(), CV_8UC1);
    double min, max;
    minMaxLoc(input_image, &min, &max);

    for (int i = 0; i < output_image.rows; i++) {
        for (int j = 0; j < output_image.cols; j++) {
            double slope = (255 - 0) / (max - min);
            double output = 0 + slope * (input_image.at<uchar>(i, j) - min);
            output_image.at<uchar>(i, j) = uchar(output);
        }
    }
    return output_image;
}

/**
 * @brief Calculate the histogram of an image to get the PDF distribution of the intensities.
 *
 * @param input_image The input grayscale image
 * @param normalize  Normalize the output PDF or not;
 * @return std::vector<double>
 */
std::vector<double> calculateHistogram(Mat input_image, bool normalize)
{
    std::vector<double> histogram(256, 0.0);

    for (int i = 0; i < input_image.rows; i++) {
        for (int j = 0; j < input_image.cols; j++) {
            histogram[int(input_image.at<uchar>(i, j))] += 1;
        }
    }

    if (normalize) {
        for (int i = 0; i < histogram.size(); i++) {
            histogram[i] /= input_image.rows * input_image.cols;
        }
    }

    return histogram;
}
int main(int argc, char** argv)
{
    Mat image = imread("../resources/lena.png", -1);
    // grayscale
    Mat grayscale = convertGrayscale(image);
    // contrast strech the image
    Mat output_image = contrastStreching(grayscale);
    namedWindow("Display Image", WINDOW_AUTOSIZE);
    imshow("Display Image", output_image);
    waitKey(0);
    return 0;
}