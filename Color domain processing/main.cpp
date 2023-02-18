#include "histogram_processing.hpp"
#include <cmath>
#include <opencv2/opencv.hpp>
#include <stdio.h>
using namespace cv;

/**
 * @brief Convert an image from the BGR color space(default OpenCV) to HSI.
 *
 * @param input_image BGR color space image
 * @return Mat
 */
Mat rgbToHSI(Mat input_image)
{
    Mat output_image = Mat(input_image.size(), CV_8UC3);

    for (int i = 0; i < input_image.rows; i++) {
        for (int j = 0; j < input_image.cols; j++) {
            double R = input_image.at<Vec3b>(i, j)[2] / 255.0;
            double G = input_image.at<Vec3b>(i, j)[1] / 255.0;
            double B = input_image.at<Vec3b>(i, j)[0] / 255.0;
            double intensity = (R + G + B) / 3.0;
            double saturation = 1 - ((3 * min(min(R, G), B)) / (R + G + B));
            double hue_degrees;
            if (saturation != 0) {
                double num = 0.5 * ((R - G) + (R - B));
                double den = sqrt((R - G) * (R - G) + (R - B) * (G - B));
                double hue = acos(num / den);
                hue_degrees = hue * (180.0 / M_PI);
                if (B > G) {
                    hue_degrees = 360 - hue_degrees;
                }
            } else {
                hue_degrees = 0;
            }

            output_image.at<Vec3b>(i, j) = Vec3b(hue_degrees / 2, saturation * 255, intensity * 255);
        }
    }
    return output_image;
}

/**
 * @brief Convert an image from the HSI color space to RGB color space
 *
 * @param input_image HSI color space image
 * @return Mat
 */
Mat hsiToRGB(Mat input_image)
{
    Mat output_image = Mat(input_image.size(), CV_8UC3);

    for (int i = 0; i < input_image.rows; i++) {
        for (int j = 0; j < input_image.cols; j++) {
            double hue_value = (input_image.at<Vec3b>(i, j)[0]) * 2;
            double saturation_value = (input_image.at<Vec3b>(i, j)[1]) / 255.0;
            double intensity_value = (input_image.at<Vec3b>(i, j)[2]) / 255.0;

            double rad_hue = hue_value * (M_PI / 180.0);

            double blue_value;
            double red_value;
            double green_value;

            if (hue_value < 120) {
                blue_value = intensity_value * (1 - saturation_value);
                red_value = intensity_value * (1 + ((saturation_value * cos(rad_hue))) / cos(1.0471975512 - rad_hue));
                green_value = 3 * intensity_value - (red_value + blue_value);
            } else if (hue_value >= 120 && hue_value <= 240) {
                rad_hue = rad_hue - 2 * M_PI / 3;
                red_value = intensity_value * (1 - saturation_value);
                green_value = intensity_value * (1 + ((saturation_value * cos(rad_hue))) / cos(1.0471975512 - rad_hue));
                blue_value = 3 * intensity_value - (red_value + green_value);
            } else {
                rad_hue = rad_hue - 4 * M_PI / 3;
                green_value = intensity_value * (1 - saturation_value);
                blue_value = intensity_value * (1 + ((saturation_value * cos(rad_hue))) / cos(1.0471975512 - rad_hue));
                red_value = 3 * intensity_value - (blue_value + green_value);
            }

            uchar blue_value_final = blue_value * 255;
            uchar red_value_final = red_value * 255;
            uchar green_value_final = green_value * 255;
            output_image.at<Vec3b>(i, j) = Vec3b(blue_value_final, green_value_final, red_value_final);
        }
    }

    return output_image;
}

/**
 * @brief Slice a grayscale image into 3 colors given the plane slice alghorithm
 *
 * @param input_image the grayscale input mat
 * @return Mat
 */
Mat planeSlice(Mat input_image)
{
    Mat output_image = Mat(input_image.size(), CV_8UC3);
    int slice_size = 255.0 / 3;
    for (int i = 0; i < input_image.rows; i++) {
        for (int j = 0; j < input_image.cols; j++) {
            if (input_image.at<uchar>(i, j) < slice_size) {
                output_image.at<Vec3b>(i, j) = Vec3b(255, 0, 0);
            } else if (input_image.at<uchar>(i, j) > slice_size && input_image.at<uchar>(i, j) < 2 * slice_size) {
                output_image.at<Vec3b>(i, j) = Vec3b(0, 255, 0);
            } else {
                output_image.at<Vec3b>(i, j) = Vec3b(0, 0, 255);
            }
        }
    }
    return output_image;
}

/**
 * @brief Complement of the RGB image.
 *
 * @param input_image
 * @return Mat
 */
Mat complemenetRGB(Mat input_image)
{
    Mat output_image = Mat(input_image.size(), CV_8UC3);

    for (int i = 0; i < input_image.rows; i++) {
        for (int j = 0; j < input_image.cols; j++) {
            output_image.at<Vec3b>(i, j) = Vec3b(255 - input_image.at<Vec3b>(i, j)[0], 255 - input_image.at<Vec3b>(i, j)[1], 255 - input_image.at<Vec3b>(i, j)[2]);
        }
    }

    return output_image;
}

/**
 * @brief Slice an RGB image given a cube subspace denoting the desired color to slice.
 *
 * @param input_image The RGB input image
 * @param desired_color_center The center of the sub-cube of the RGB color space which contains the desired colors.
 * @param width The width of the cube.
 * @return Mat
 */
Mat colorSliceRGB(Mat input_image, Vec3b desired_color_center, int width)
{
    Mat output_image = Mat(input_image.size(), CV_8UC3);

    for (int i = 0; i < input_image.rows; i++) {
        for (int j = 0; j < input_image.cols; j++) {
            if (abs(input_image.at<Vec3b>(i, j)[0] - desired_color_center[2]) <= (width / 2.0) && abs(input_image.at<Vec3b>(i, j)[1] - desired_color_center[1]) <= (width / 2.0) && abs(input_image.at<Vec3b>(i, j)[2] - desired_color_center[0]) <= (width / 2.0)) {
                output_image.at<Vec3b>(i, j) = Vec3b(255, 255, 255);
            } else {
                output_image.at<Vec3b>(i, j) = Vec3b(0, 0, 0);
            }
        }
    }

    return output_image;
}
/**
 * @brief Slice the color of an image given the HSI color model.
 *
 * @param input_image Ths HSI input image.
 * @param lower_bound The lower bound of hue degrees.
 * @param upper_bound The upper hbound of hue degrees.
 * @return Mat
 */
Mat colorSliceHSI(Mat input_image, int lower_bound, int upper_bound)
{
    Mat output_image = Mat(input_image.size(), CV_8UC3);

    for (int i = 0; i < input_image.rows; i++) {
        for (int j = 0; j < input_image.cols; j++) {
            int hue = input_image.at<Vec3b>(i, j)[0];
            if (hue >= lower_bound && hue <= upper_bound) {
                output_image.at<Vec3b>(i, j) = Vec3b(255, 255, 255);
            } else {
                output_image.at<Vec3b>(i, j) = Vec3b(0, 0, 0);
            }
        }
    }

    return output_image;
}

/**
 * @brief Equalize an image based on the HSI color model by linearly changing the slope of the current intensity interval to the max possible [0,255]
 *
 * @param input_image The RGB input image
 * @return Mat
 */
Mat intensityEqualization(Mat input_image)
{
    Mat output_image = Mat(input_image.size(), CV_8UC3);
    // Split the channels and extract the intensity
    std::vector<Mat> channels(3);
    split(input_image, channels);
    Mat intensity_channel = channels[2];
    // Get min and max from the intensity channel
    double min_val, max_val;
    minMaxIdx(intensity_channel, &min_val, &max_val);
    double min_range = 0;
    double max_range = 255;

    for (int i = 0; i < intensity_channel.rows; i++) {
        for (int j = 0; j < intensity_channel.cols; j++) {
            double result_intensity = min_range + ((max_range - min_range) / (max_val - min_val)) * (intensity_channel.at<uchar>(i, j) - (min_val + 5));
            output_image.at<Vec3b>(i, j) = Vec3b(input_image.at<Vec3b>(i, j)[0], input_image.at<Vec3b>(i, j)[1], (uchar)(result_intensity));
        }
    }

    return output_image;
}

/**
 * @brief Equalize a color image (in the HSI color space) using the histogram method/
 *
 * @param input_image HSI color space image.
 * @return Mat
 */
Mat histogramEqualization(Mat input_image)
{
    Mat output_image = Mat(input_image.size(), CV_8UC3);
    // Create the desired histogram
    std::vector<double> desired_histogram(256, 0.0);
    for (int i = 0; i < desired_histogram.size(); i++) {
        desired_histogram[i] = float(1) / 255.0;
    }
    // Split the channels and extract the intensity one
    std::vector<Mat> channels(3);
    split(input_image, channels);
    Mat intensity_channel = channels[2];

    Mat intensity_output = matchHistogram(intensity_channel, desired_histogram);

    for (int i = 0; i < input_image.rows; i++) {
        for (int j = 0; j < input_image.cols; j++) {
            // the problem is that the intensity is going overboard kinda
            uchar final_intensity = intensity_output.at<uchar>(i, j);
            output_image.at<Vec3b>(i, j) = Vec3b(input_image.at<Vec3b>(i, j)[0], input_image.at<Vec3b>(i, j)[1], final_intensity - 42);
        }
    }

    return output_image;
}

Mat colorKernel(Mat input_image, Mat kernel)
{
    Mat output_image = Mat(input_image.size(), CV_8UC3);

    Vec3b sum_kernel = Vec3b(sum(kernel)[0], sum(kernel)[0], sum(kernel)[0]);

    for (int i = kernel.rows / 2; i < input_image.rows - (kernel.rows / 2); i++) {
        for (int j = kernel.rows / 2; j < input_image.cols - (kernel.rows / 2); j++) {
            Mat neighbourhood = Mat(kernel.size(), CV_8UC3);
            for (int k = 0; k < kernel.rows * kernel.cols; k++) {
                int current_neighbourhood_i = i + (k / kernel.rows) - (kernel.rows / 2);
                int current_neighbourhood_j = j + (k % kernel.rows) - (kernel.rows / 2);
                neighbourhood.at<Vec3b>(k / kernel.rows, k % kernel.rows) = input_image.at<Vec3b>(current_neighbourhood_i, current_neighbourhood_j);
            }
            Mat result_convolution = Mat(kernel.size(), CV_32FC3);
            result_convolution = neighbourhood.mul(kernel);
            Vec3f result = Vec3f(sum(result_convolution)[0], sum(result_convolution)[1], sum(result_convolution)[2]);
            output_image.at<Vec3b>(i, j) = Vec3b(result[0] / sum_kernel[0], result[1] / sum_kernel[1], result[2] / sum_kernel[2]);
        }
    }

    return output_image;
}

int main(int argc, char** argv)
{
    Mat image = imread("../resources/lena.png", 1);
    Mat hsi_image = rgbToHSI(image);
    Mat gray;
    cvtColor(image, gray, COLOR_BGR2GRAY);
    // Mat complement_rgb = complemenetRGB(image);
    // Mat hsi_equalized = intensityEqualization(hsi_image);
    // Mat rgb_equalized = hsiToRGB(hsi_equalized);

    Mat kernel = Mat(3, 3, CV_8UC3, Vec3b(1, 1, 1));
    Mat hsi_equlaized = histogramEqualization(hsi_image);
    Mat rgb_equalized = hsiToRGB(hsi_equlaized);
    Mat smoothening_color = colorKernel(image, kernel);
    imshow("Display Image1", image);
    imshow("Display Image2", smoothening_color);
    waitKey(0);
}