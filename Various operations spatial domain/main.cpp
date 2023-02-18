#include <cmath>
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

/**
 * @brief Calculate the cummulative histogram of an image
 *
 * @param input_image The input image
 * @return std::vector<double>
 */
std::vector<int> calculateCummulativeHistogram(Mat input_image)
{
    std::vector<double> input_histogram = calculateHistogram(input_image, true);
    std::vector<double> input_cummulative_histogram(256, 0.0);
    std::vector<int> result(256, 0); // result must be an integer
    for (int i = 0; i < input_histogram.size(); i++) {
        for (int j = 0; j <= i; j++) {
            input_cummulative_histogram[i] += input_histogram[j];
        }
        result[i] = round(255 * input_cummulative_histogram[i]);
    }

    return result;
}

/**
 * @brief Calculate the cummulative histogram of an image
 *
 * @param input_histogram The input histogram
 * @return std::vector<double>
 */
std::vector<int> calculateCummulativeHistogram(std::vector<double> input_histogram)
{
    std::vector<double> input_cummulative_histogram(256, 0.0);
    std::vector<int> result(256, 0); // result must be an integer
    for (int i = 0; i < input_histogram.size(); i++) {
        for (int j = 0; j <= i; j++) {
            input_cummulative_histogram[i] += input_histogram[j];
        }
        result[i] = round(255 * input_cummulative_histogram[i]);
    }

    return result;
}

/**
 * @brief Apply an intensity transform based on the gamma power law
 *
 * @param input_image Input image
 * @param gamma Power of the gamma function applied
 * @return Mat
 */
Mat gammaPower(Mat input_image, double gamma)
{
    Mat output_image = Mat(input_image.size(), CV_8UC1);

    for (int i = 0; i < input_image.rows; i++) {
        for (int j = 0; j < input_image.cols; j++) {
            double normalized_input = double(input_image.at<uchar>(i, j)) / 255;
            uchar result = uchar(pow(normalized_input, gamma) * 255);
            output_image.at<uchar>(i, j) = result;
        }
    }

    return output_image;
}

/**
 * @brief Modify an image based on a given expected histogram.
 *
 * @param input_image
 * @param specified_histogram
 * @return Mat
 */
Mat histogramSpecification(Mat input_image, std::vector<double> specified_histogram)
{
    Mat output_image = Mat(input_image.size(), CV_8UC1);
    std::vector<double> input_histogram = calculateHistogram(input_image, true);

    // calculate the cummulative distribution function of input image
    std::vector<int> cummulative_input_histogram = calculateCummulativeHistogram(input_image);
    // calculate the cummulative distribution of the specified histogram
    std::vector<int> cummulative_expected_histogram = calculateCummulativeHistogram(specified_histogram);
    std::vector<int> matching_histogram(256, 0);
    for (int i = 0; i < cummulative_input_histogram.size(); i++) { // for each s_k value
        int best_match = 999;
        for (int j = 0; j < cummulative_expected_histogram.size(); j++) {
            if (std::find(matching_histogram.begin(), matching_histogram.end(), cummulative_expected_histogram[j]) == matching_histogram.end()) {
                if (abs(cummulative_input_histogram[i] - cummulative_expected_histogram[j]) < best_match) {
                    best_match = abs(cummulative_input_histogram[i] - cummulative_expected_histogram[j]);
                    matching_histogram[i] = cummulative_expected_histogram[j];
                }
            }
        }
        if (std::find(matching_histogram.begin(), matching_histogram.end(), 255) != matching_histogram.end()) { // no match
            matching_histogram[i] = 255;
        }
    }

    for (int i = 0; i < output_image.rows; i++) {
        for (int j = 0; j < output_image.cols; j++) {
            output_image.at<uchar>(i, j) = matching_histogram[input_image.at<uchar>(i, j)];
        }
    }
    return output_image;
}

/**
 * @brief Apply a given kernel to an image
 *
 * @param input_image The input image ( must be grayscale)
 * @param filter The applied filter
 * @param convolve If true convolution will rotate the filter 180* ; otherwise the correlation process proceeds.
 * @return Mat
 */
Mat applyKernel(Mat input_image, Mat filter, bool convolve)
{
    Mat intermediate_image = Mat::zeros(input_image.size(), CV_32F);
    Mat output_image = Mat(input_image.size(), CV_8UC1);
    // Make the distinction between convolution and correlation.
    Mat filter_rotated;
    uchar filter_sum = sum(filter)[0]; // rotation does not affect it
    if (filter_sum == 0) {
        filter_sum = 1;
    }
    if (convolve) {
        rotate(filter, filter_rotated, ROTATE_180);
    } else {
        filter_rotated = filter;
    }

    for (int i = 1; i < input_image.rows - 1; i++) {
        for (int j = 1; j < input_image.cols - 1; j++) {
            for (int k = 0; k < filter_rotated.rows * filter_rotated.cols; k++) {
                float x = (input_image.at<uchar>(i + (k / filter_rotated.rows) - 1, j + (k % filter_rotated.rows) - 1) * filter_rotated.at<float>(k / filter_rotated.rows, k % filter_rotated.rows));
                intermediate_image.at<float>(i, j) += (input_image.at<uchar>(i + (k / filter_rotated.rows) - 1, j + (k % filter_rotated.rows) - 1) * filter_rotated.at<float>(k / filter_rotated.rows, k % filter_rotated.rows));
            }
            // convert the resulting values to uchar
            if (intermediate_image.at<float>(i, j) < 0) {
                output_image.at<uchar>(i, j) = 0;
            } else {
                output_image.at<uchar>(i, j) = uchar(intermediate_image.at<float>(i, j));
            }
        }
    }

    return output_image;
}

/**
 * @brief Apply the gaussian filter to an image
 *
 * @param input_image
 * @param k_size the kernel size; given by [k_size x k_size]
 * @param variance the variance of the normal distribution
 * @return Mat
 */
Mat gaussFilter(Mat input_image, int k_size, int variance)
{
    // Define the guassian filter given the size
    Mat gaussian_kernel = Mat(k_size, k_size, CV_32F);
    for (int k = 0; k < k_size * k_size; k++) {
        double current_neighbourhood_i = (k / k_size) - (k_size / 2);
        double current_neighbourhood_j = (k % k_size) - (k_size / 2);
        double distance_from_center = sqrt((current_neighbourhood_i * current_neighbourhood_i) + (current_neighbourhood_j * current_neighbourhood_j));
        gaussian_kernel.at<float>(k / k_size, k % k_size) = exp((-1 * distance_from_center * distance_from_center) / (2 * variance * variance));
    }

    double kernel_sum = sum(gaussian_kernel)[0];
    // Apply the resulting kernel
    Mat output_image = Mat(input_image.size(), CV_8UC1);
    for (int i = k_size; i < input_image.rows - k_size; i++) {
        for (int j = k_size; j < input_image.cols - k_size; j++) {
            float intermediate_value = 0;
            for (int k = 0; k < k_size * k_size; k++) {
                intermediate_value += (input_image.at<uchar>(i + (k / k_size) - k_size / 2, j + (k % k_size) - k_size / 2) * gaussian_kernel.at<float>(k / k_size, k % k_size));
            }
            float intermediate_sum = sum(gaussian_kernel)[0];
            int result = intermediate_value / intermediate_sum;
            output_image.at<uchar>(i, j) = uchar(intermediate_value / intermediate_sum);
        }
    }

    return output_image;
}

/**
 * @brief Median filter of an image.
 *
 * @param input_image The input image
 * @param k_size the kernel size
 * @return Mat
 */
Mat medianFilter(Mat input_image, int k_size)
{
    Mat output_image = Mat(input_image.size(), CV_8UC1);
    // Define the kernel
    std::vector<uchar> kernel;
    for (int i = 0; i < input_image.rows; i++) {
        for (int j = 0; j < input_image.cols; j++) {
            for (int k = 0; k < k_size * k_size; k++) {
                kernel.push_back(input_image.at<uchar>(i + (k / k_size) - 1, j + (k % k_size) - 1));
            }
            std::sort(kernel.begin(), kernel.end());
            output_image.at<uchar>(i, j) = kernel.at(int(kernel.size() / 2));
            kernel.clear();
        }
    }

    return output_image;
}

/**
 * @brief Boost the sharp parts of an image using a smoothened mask.
 *
 * @param input_image The input image
 * @param k The ratio of the mask
 * @return Mat
 */
Mat unsharpBoosting(Mat input_image, int k)
{
    Mat output_image;
    Mat blurred_image;
    GaussianBlur(input_image, blurred_image, Size(3, 3), 0);
    Mat mask;
    subtract(input_image, blurred_image, mask);
    Mat enhanced_mask = mask * k;
    add(input_image, enhanced_mask, output_image);

    return output_image;
}

int main(int argc, char** argv)
{
    Mat image = imread("../resources/lena.png", -1);
    // grayscale
    Mat grayscale = convertGrayscale(image);
    // contrast strech the image
    // Mat output_image = contrastStreching(grayscale);
    // // Gamma power
    // Mat gamme_image = gammaPower(grayscale, 3);
    // // Equalized histogram
    // std::vector<double> desired_histogram(256, 0.0);
    // for (int i = 0; i < desired_histogram.size(); i++) {
    //     desired_histogram[i] = float(1) / (255);
    // }
    // Mat out = histogramSpecification(grayscale, desired_histogram);
    // namedWindow("Display Image", WINDOW_AUTOSIZE);
    float filter_array[3][3] = {
        { -1.0, 0.0, 1.0 },
        { -2.0, 0.0, 2.0 },
        { -1.0, 0.0, 1.0 }
    };
    Mat filter = Mat(3, 3, CV_32F, &filter_array);
    Mat kernel_input;
    // GaussianBlur(grayscale, kernel_input, Size(3, 3), 0, 0, BORDER_DEFAULT);
    // Mat result = applyKernel(kernel_input, filter, false);
    // Mat gaussian_image = gaussFilter(grayscale, 3, 1);
    // Mat median_image = medianFilter(grayscale, 3);
    Mat blurred_image = unsharpBoosting(grayscale, 3);
    imshow("Display Image", blurred_image);
    imshow("Display Image2", grayscale);
    waitKey(0);
    return 0;
}