#include <cmath>
#include <opencv2/opencv.hpp>
#include <stdio.h>

using namespace cv;

Mat padImage(Mat input_image, int pad_size)
{
    Mat padded;
    padded.create(input_image.rows + pad_size, input_image.cols + pad_size, input_image.type());
    padded.setTo(cv::Scalar::all(255));

    input_image.copyTo(padded(Rect(pad_size, pad_size, input_image.cols, input_image.rows)));

    return padded;
}

/**
 * @brief Threshold a image given a fixed value.
 *
 * @param input_image The input image
 * @param threshold_val the threshold value
 * @return Mat
 */
Mat valueThresholding(Mat input_image, int threshold_val)
{
    Mat output_image = Mat(input_image.size(), CV_8U);

    for (int i = 0; i < input_image.rows; i++) {
        for (int j = 0; j < input_image.cols; j++) {
            output_image.at<uchar>(i, j) = input_image.at<uchar>(i, j) > threshold_val ? 255 : 0;
        }
    }

    return output_image;
}

/**
 * @brief Threshold an image given the local mean.
 *
 * @param input_image The input image
 * @param k_size The neighbourhood size [k_size x k_size]
 * @param c the constant substracted from the output value
 * @return Mat
 */
Mat meanThresholding(Mat input_image, int k_size, int c)
{
    Mat kernel = Mat(k_size, k_size, CV_32FC1, 1.0);
    Mat output_image = Mat(input_image.size(), CV_8U);

    for (int i = k_size / 2; i < input_image.rows - k_size / 2; i += 1) {
        for (int j = k_size / 2; j < input_image.cols - k_size / 2; j += 1) {
            float mean_sum = 0;
            for (int k = 0; k < k_size * k_size; k++) {
                mean_sum += input_image.at<uchar>(i + (k / k_size) - k_size / 2, j + (k % k_size) - k_size / 2) * kernel.at<float>(k / k_size, k % k_size);
            }
            float mean = mean_sum / (k_size * k_size) - c;
            output_image.at<uchar>(i, j) = input_image.at<uchar>(i, j) > mean ? 255 : 0;
        }
    }

    return output_image;
}

/**
 * @brief Threshold an image given the local gaussian function.
 *
 * @param input_image The input image
 * @param k_size The neighbourhood size [k_size x k_size]
 * @param c the constant substracted from the output value
 * @param variance the Variance of the distribution created from the local neighbourhood
 * @return Mat
 */
Mat gaussianThresholding(Mat input_image, int k_size, int c, int variance)
{
    Mat output_image = Mat(input_image.size(), CV_8U);
    // Define and create a gaussian distribution based on the kernel size
    Mat kernel = Mat(k_size, k_size, CV_32FC1, 0.0);
    for (int k = 0; k < k_size * k_size; k++) {
        double current_neighbourhood_i = (k / k_size) - (k_size / 2);
        double current_neighbourhood_j = (k % k_size) - (k_size / 2);
        double distance_from_center = sqrt((current_neighbourhood_i * current_neighbourhood_i) + (current_neighbourhood_j * current_neighbourhood_j));
        kernel.at<float>(k / k_size, k % k_size) = exp((-1 * distance_from_center * distance_from_center) / (2 * variance * variance));
    }
    double kernel_sum = sum(kernel)[0];

    // Given the gaussian kernel Apply a threshold
    for (int i = k_size / 2; i < input_image.rows - k_size / 2; i += 1) {
        for (int j = k_size / 2; j < input_image.cols - k_size / 2; j += 1) {
            float mean_sum = 0;
            for (int k = 0; k < k_size * k_size; k++) {
                mean_sum += input_image.at<uchar>(i + (k / k_size) - k_size / 2, j + (k % k_size) - k_size / 2) * kernel.at<float>(k / k_size, k % k_size);
            }
            float gaussian_mean = (mean_sum / kernel_sum) - c;
            output_image.at<uchar>(i, j) = input_image.at<uchar>(i, j) > gaussian_mean ? 255 : 0;
        }
    }

    return output_image;
}

int main(int argc, char** argv)
{
    Mat image = imread("../resources/sudokubig.jpg", -1);
    Mat grayscale;
    cvtColor(image, grayscale, COLOR_BGR2GRAY);
    Mat blurred_grayscale;
    medianBlur(grayscale, blurred_grayscale, 5);
    Mat threshold = valueThresholding(blurred_grayscale, 127);
    Mat threshold_mean = meanThresholding(blurred_grayscale, 11, 2);
    Mat threshold_gaussian = gaussianThresholding(blurred_grayscale, 11, 2, 2);
    Mat thres_cv;
    adaptiveThreshold(blurred_grayscale, thres_cv, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 11, 2);
    imshow("Display Image", blurred_grayscale);
    imshow("Result opencv", thres_cv);
    imshow("Result me", threshold_gaussian);
    waitKey(0);
    return 0;
}