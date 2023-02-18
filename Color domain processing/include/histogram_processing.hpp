#include <cmath>
#include <opencv2/opencv.hpp>
#include <stdio.h>

using namespace cv;

std::vector<double> computeHistogram(Mat input_image)
{
    std::vector<double> histogram(256, 0.0);

    for (int i = 0; i < input_image.rows; i++) {
        for (int j = 0; j < input_image.cols; j++) {
            histogram[input_image.at<uchar>(i, j)] += 1;
        }
    }

    for (int i = 0; i < histogram.size(); i++) {
        histogram[i] = histogram[i] / float((input_image.rows * input_image.cols));
    }

    return histogram;
}

std::vector<int> computeCummulativeHistogram(std::vector<double> input)
{
    std::vector<int> cummulative_histogram(256, 0.0);

    for (int i = 0; i < input.size(); i++) {
        double sum = 0;
        for (int j = 0; j <= i; j++) {
            sum += input[j];
        }
        cummulative_histogram[i] = round(256 * sum);
    }

    return cummulative_histogram;
}

Mat matchHistogram(Mat input_image, std::vector<double> specified_histogram)
{
    Mat output_image = Mat(input_image.size(), CV_8UC1);
    std::vector<double> input_histogram = computeHistogram(input_image);

    std::vector<int> cummulative_input_histogram = computeCummulativeHistogram(input_histogram);
    std::vector<int> cummulative_expected_histogram = computeCummulativeHistogram(specified_histogram);

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