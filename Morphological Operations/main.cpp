#include <cmath>
#include <opencv2/opencv.hpp>
#include <stdio.h>

using namespace cv;

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
            output_image.at<uchar>(i, j) = input_image.at<uchar>(i, j) > gaussian_mean ? 0 : 255;
        }
    }

    return output_image;
}

/**
 * @brief Threshold an image given a scalar value.
 *
 * @param input_image Grayscale input image
 * @param value the value along which to binarize
 * @return Mat
 */
Mat thresholdValue(Mat input_image, int value)
{
    Mat output_image = Mat(input_image.size(), CV_8UC1);

    for (int i = 0; i < input_image.rows; i++) {
        for (int j = 0; j < input_image.cols; j++) {
            if (input_image.at<uchar>(i, j) < value) {
                output_image.at<uchar>(i, j) = 0;
            } else {
                output_image.at<uchar>(i, j) = 255;
            }
        }
    }

    return output_image;
}

/**
 * @brief Apply an erosion morphological operation to the input image.
 *
 * @param input_image The binary input image
 * @param structural_element Mat denoting the structural element
 * @return Mat
 */
Mat erosion(Mat input_image, Mat structural_element, uchar foreground_value)
{
    //  background pixels = 255
    //  foreground pixels = 0
    Mat output_image = Mat(input_image.size(), CV_8UC1, 255 - foreground_value);

    for (int i = structural_element.rows / 2; i < input_image.rows - structural_element.rows / 2; i++) {
        for (int j = structural_element.cols / 2; j < input_image.cols - structural_element.cols / 2; j++) {
            // Create the neighbourhood of the current pixel
            Mat neighbourhood = Mat(structural_element.size(), CV_8UC1);
            for (int k = 0; k < structural_element.rows * structural_element.cols; k++) {
                int current_neighbourhood_i = i + (k / structural_element.rows) - (structural_element.rows / 2);
                int current_neighbourhood_j = j + (k % structural_element.rows) - (structural_element.rows / 2);
                neighbourhood.at<uchar>(k / structural_element.rows, k % structural_element.cols) = input_image.at<uchar>(current_neighbourhood_i, current_neighbourhood_j);
            }
            // Convolve the neighbourhood with the structural element
            // Check if B(z) included in I .if the structural element translated by z value corresponds with what it is in the image
            bool is_included = true;
            for (int k = 0; k < structural_element.rows * structural_element.cols; k++) {
                if (structural_element.at<int>(k / structural_element.rows, k % structural_element.cols) == foreground_value) { // only analyse background values
                    if (structural_element.at<int>(k / structural_element.rows, k % structural_element.cols) != neighbourhood.at<uchar>(k / structural_element.rows, k % structural_element.cols)) {
                        is_included = false;
                        break;
                    }
                }
            }
            if (is_included) {
                output_image.at<uchar>(i, j) = foreground_value;
            } else {
                output_image.at<uchar>(i, j) = 255 - foreground_value;
            }
        }
    }

    return output_image;
}

/**
 * @brief Apply a dilation morphological operation to the input image.
 *
 * @param input_image The binary input image
 * @param structural_element Mat denoting the structural element
 * @return Mat
 */
Mat dilation(Mat input_image, Mat structural_element, uchar foreground_value)
{
    //  background pixels = 255
    //  foreground pixels = 0
    Mat output_image = Mat(input_image.size(), CV_8UC1, 255 - foreground_value);

    for (int i = structural_element.rows / 2; i < input_image.rows - structural_element.rows / 2; i++) {
        for (int j = structural_element.cols / 2; j < input_image.cols - structural_element.cols / 2; j++) {
            // Create the neighbourhood of the current pixel
            Mat neighbourhood = Mat(structural_element.size(), CV_8UC1);
            Mat and_result = Mat(structural_element.size(), CV_32SC1);
            for (int k = 0; k < structural_element.rows * structural_element.cols; k++) {
                int current_neighbourhood_i = i + (k / structural_element.rows) - (structural_element.rows / 2);
                int current_neighbourhood_j = j + (k % structural_element.rows) - (structural_element.rows / 2);
                neighbourhood.at<uchar>(k / structural_element.rows, k % structural_element.cols) = input_image.at<uchar>(current_neighbourhood_i, current_neighbourhood_j);
            }
            // Convolve the neighbourhood with the structural element
            int set_sum = sum(neighbourhood)[0];
            // If it is not full background (255) then we have a foreground pixel in the structuring element => dilate
            if (set_sum != ((255 - foreground_value) * structural_element.rows * structural_element.cols)) {
                output_image.at<uchar>(i, j) = foreground_value;
            } else {
                output_image.at<uchar>(i, j) = 255 - foreground_value;
            }
        }
    }

    return output_image;
}

/**
 * @brief Open the image with the given structuring element.
 *
 * @param input_image  binary image
 * @param structuring_element
 * @return Mat
 */
Mat open(Mat input_image, Mat structuring_element, uchar foreground_value)
{
    Mat erosion_ = erosion(input_image, structuring_element, foreground_value);
    Mat dilation_ = dilation(erosion_, structuring_element, foreground_value);

    return dilation_;
}

/**
 * @brief Close the image with the given structuring element.
 *
 * @param input_image binary image
 * @param structuriung_element
 * @return Mat
 */
Mat close(Mat input_image, Mat structuriung_element, uchar foreground_value)
{
    Mat dilation_ = dilation(input_image, structuriung_element, foreground_value);
    Mat erosion_ = erosion(dilation_, structuriung_element, foreground_value);

    return erosion_;
}

/**
 * @brief Complement the binary image.
 *
 * @param input_image
 * @return Mat
 */
Mat complement_image(Mat input_image)
{
    Mat output_image = Mat(input_image.size(), CV_8UC1);

    for (int i = 0; i < input_image.rows; i++) {
        for (int j = 0; j < input_image.cols; j++) {
            if (input_image.at<uchar>(i, j) == 255) {
                output_image.at<uchar>(i, j) = 0;
            } else {
                output_image.at<uchar>(i, j) = 255;
            }
        }
    }

    return output_image;
}
/**
 * @brief Perform a hit or miss operation given the set op.
 *
 * @param input_image Binary input image
 * @param structuring_element_foreground se used to detect foreground
 * @param structuring_element_background se used to detect foreground's border
 * @return Mat
 */
Mat hit_or_miss(Mat input_image, Mat structural_element, uchar foreground_value)
{
    //  background pixels = 255
    //  foreground pixels = 0
    Mat output_image = Mat(input_image.size(), CV_8UC1, 255 - foreground_value);

    for (int i = structural_element.rows / 2; i < input_image.rows - structural_element.rows / 2; i++) {
        for (int j = structural_element.cols / 2; j < input_image.cols - structural_element.cols / 2; j++) {
            // Create the neighbourhood of the current pixel
            Mat neighbourhood = Mat(structural_element.size(), CV_8UC1);
            for (int k = 0; k < structural_element.rows * structural_element.cols; k++) {
                int current_neighbourhood_i = i + (k / structural_element.rows) - (structural_element.rows / 2);
                int current_neighbourhood_j = j + (k % structural_element.rows) - (structural_element.rows / 2);
                neighbourhood.at<uchar>(k / structural_element.rows, k % structural_element.cols) = input_image.at<uchar>(current_neighbourhood_i, current_neighbourhood_j);
            }
            // Convolve the neighbourhood with the structural element
            // Check if B(z) included in I .if the structural element translated by z value corresponds with what it is in the image
            bool is_included = true;
            for (int k = 0; k < structural_element.rows * structural_element.cols; k++) {
                if (structural_element.at<int>(k / structural_element.rows, k % structural_element.cols) != -1) { // no dont care value
                    if (structural_element.at<int>(k / structural_element.rows, k % structural_element.cols) != neighbourhood.at<uchar>(k / structural_element.rows, k % structural_element.cols)) {
                        is_included = false;
                        break;
                    }
                }
            }
            if (is_included) {
                output_image.at<uchar>(i, j) = foreground_value;
            } else {
                output_image.at<uchar>(i, j) = 255 - foreground_value;
            }
        }
    }

    return output_image;
}

/**
 * @brief Extract the border of foreground objects
 *
 * @param input_image Binary input image
 * @return Mat
 */
Mat border_extract(Mat input_image)
{
    // Works only for foreground 255 background 0
    Mat structuring_element = Mat(Size(3, 3), CV_8UC1, 255);
    Mat erosion_mat = erosion(input_image, structuring_element, 255);
    return input_image - erosion_mat;
}

/**
 * @brief Fill a whole of a given image using a dilation and comparison with the border in original
 *
 * @param input_image The input image
 * @param hole_point A random hole point of the object of interest
 * @return Mat
 */
Mat hole_filling(Mat input_image, Point2d hole_point)
{
    Mat output_image = Mat::zeros(input_image.size(), CV_8UC1);
    Mat last_image = Mat::zeros(input_image.size(), CV_8UC1);
    output_image.at<uchar>(hole_point.x, hole_point.y) = 255;
    Mat structuring_element = Mat(Size(3, 3), CV_8UC1, 255);
    Mat original_complement = complement_image(input_image);
    while (sum(output_image - last_image)[0] != 0) {
        last_image = output_image.clone();
        output_image = dilation(output_image, structuring_element, 255);
        bitwise_and(output_image, original_complement, output_image);
    }

    return output_image;
}

/**
 * @brief Obtain an image component defined with a containing @param component_point
 *
 * @param input_image The binary image
 * @param component_point A point included in the desired component
 * @return Mat
 */
Mat connected_component(Mat input_image, Point2d component_point)
{
    Mat output_image = Mat::zeros(input_image.size(), CV_8UC1);
    Mat last_image = Mat::zeros(input_image.size(), CV_8UC1);
    output_image.at<uchar>(component_point.x, component_point.y) = 255;
    Mat structuring_element = Mat(Size(3, 3), CV_8UC1, 255);
    while (sum(output_image - last_image)[0] != 0) {
        last_image = output_image.clone();
        output_image = dilation(output_image, structuring_element, 255);
        bitwise_and(output_image, input_image, output_image);
    }

    return output_image;
}

Mat complement_structuring_element(Mat structuring_element)
{
    Mat complement_se = Mat(structuring_element.size(), CV_32SC1, 255);

    for (int i = 0; i < structuring_element.rows; i++) {
        for (int j = 0; j < structuring_element.cols; j++) {
            if (structuring_element.at<int>(i, j) == -1) {
                complement_se.at<int>(i, j) = -1;
            } else if (structuring_element.at<int>(i, j) == 255) {
                complement_se.at<int>(i, j) = 0;
            } else {
                complement_se.at<int>(i, j) = 255;
            }
        }
    }

    return complement_se;
}

/**
 * @brief Create the convex hull of the given foreground objects
 *
 * @param input_image
 * @return Mat the threshould image
 */
Mat convex_hull(Mat input_image)
{
    Mat output_image = Mat::zeros(input_image.size(), CV_8UC1);
    // initialize the structuring elements
    Mat left_border = Mat(3, 3, CV_32SC1, 255);
    left_border.at<int>(1, 1) = 0;
    left_border.at<int>(0, 1) = -1;
    left_border.at<int>(0, 2) = -1;
    left_border.at<int>(1, 2) = -1;
    left_border.at<int>(2, 2) = -1;
    left_border.at<int>(2, 1) = -1;
    Mat up_border = Mat(3, 3, CV_32SC1, 255);
    up_border.at<int>(1, 1) = 0;
    up_border.at<int>(1, 0) = -1;
    up_border.at<int>(2, 0) = -1;
    up_border.at<int>(2, 1) = -1;
    up_border.at<int>(2, 2) = -1;
    up_border.at<int>(1, 2) = -1;
    Mat right_border = Mat(3, 3, CV_32SC1, 255);
    right_border.at<int>(1, 1) = 0;
    right_border.at<int>(0, 1) = -1;
    right_border.at<int>(0, 0) = -1;
    right_border.at<int>(1, 0) = -1;
    right_border.at<int>(2, 0) = -1;
    right_border.at<int>(2, 1) = -1;
    Mat down_border = Mat(3, 3, CV_32SC1, 255);
    down_border.at<int>(1, 1) = 0;
    down_border.at<int>(0, 0) = -1;
    down_border.at<int>(0, 1) = -1;
    down_border.at<int>(0, 2) = -1;
    down_border.at<int>(1, 2) = -1;
    down_border.at<int>(1, 0) = -1;
    std::vector<Mat> structuring_elements = { left_border, up_border, right_border, down_border };
    std::vector<Mat> results;
    for (int se = 0; se < structuring_elements.size(); se++) {
        Mat last_image = input_image.clone();
        Mat output_image = hit_or_miss(input_image, structuring_elements[se], 255);
        bitwise_or(output_image, last_image, output_image);

        while (sum(output_image - last_image)[0] != 0) {
            last_image = output_image.clone();
            output_image = hit_or_miss(output_image, structuring_elements[se], 255);
            bitwise_or(output_image, last_image, output_image);
        }

        results.push_back(output_image);
    }
    Mat result_image = Mat(input_image.size(), CV_8UC1);
    bitwise_or(results.at(0), results.at(1), result_image);
    bitwise_or(results.at(2), result_image, result_image);
    bitwise_or(results.at(3), result_image, result_image);

    return result_image;
}

/**
 * @brief Thin the foreground objects using HMT transform
 *
 * @param input_image the binary image (255 foreground)
 * @param iterations number of iterations of thinning on each side
 * @return Mat
 */
Mat object_thining(Mat input_image, int iterations)
{
    Mat output_image = input_image.clone();
    // Create the structuring elements for thinning/thickening
    Mat down_thickness = Mat(3, 3, CV_32SC1, 255);
    down_thickness.at<int>(0, 0) = 0;
    down_thickness.at<int>(0, 1) = 0;
    down_thickness.at<int>(0, 2) = 0;
    down_thickness.at<int>(1, 0) = -1;
    down_thickness.at<int>(1, 2) = -1;
    Mat down_left_thickness = Mat(3, 3, CV_32SC1, 255);
    down_left_thickness.at<int>(1, 2) = 0;
    down_left_thickness.at<int>(0, 1) = 0;
    down_left_thickness.at<int>(0, 2) = 0;
    down_left_thickness.at<int>(0, 0) = -1;
    down_left_thickness.at<int>(2, 2) = -1;
    Mat left_thickness = Mat(3, 3, CV_32SC1, 255);
    left_thickness.at<int>(2, 0) = 0;
    left_thickness.at<int>(2, 1) = 0;
    left_thickness.at<int>(2, 2) = 0;
    left_thickness.at<int>(0, 1) = -1;
    left_thickness.at<int>(2, 1) = -1;
    Mat up_left_thickness = Mat(3, 3, CV_32SC1, 255);
    up_left_thickness.at<int>(1, 2) = 0;
    up_left_thickness.at<int>(2, 1) = 0;
    up_left_thickness.at<int>(2, 2) = 0;
    up_left_thickness.at<int>(2, 0) = -1;
    up_left_thickness.at<int>(0, 2) = -1;
    Mat up_thickness = Mat(3, 3, CV_32SC1, 255);
    up_thickness.at<int>(2, 0) = 0;
    up_thickness.at<int>(2, 1) = 0;
    up_thickness.at<int>(2, 2) = 0;
    up_thickness.at<int>(1, 0) = -1;
    up_thickness.at<int>(1, 2) = -1;
    Mat up_right_thickness = Mat(3, 3, CV_32SC1, 255);
    up_right_thickness.at<int>(1, 0) = 0;
    up_right_thickness.at<int>(2, 0) = 0;
    up_right_thickness.at<int>(2, 1) = 0;
    up_right_thickness.at<int>(0, 0) = -1;
    up_right_thickness.at<int>(2, 2) = -1;
    Mat right_thickness = Mat(3, 3, CV_32SC1, 255);
    right_thickness.at<int>(0, 0) = 0;
    right_thickness.at<int>(1, 0) = 0;
    right_thickness.at<int>(2, 0) = 0;
    right_thickness.at<int>(0, 1) = -1;
    right_thickness.at<int>(2, 1) = -1;
    Mat down_right_thickness = Mat(3, 3, CV_32SC1, 255);
    down_right_thickness.at<int>(0, 0) = 0;
    down_right_thickness.at<int>(0, 1) = 0;
    down_right_thickness.at<int>(1, 0) = 0;
    down_right_thickness.at<int>(0, 2) = -1;
    down_right_thickness.at<int>(2, 0) = -1;
    std::vector<Mat> structuring_elements { down_thickness, down_left_thickness, left_thickness, up_left_thickness, up_thickness, up_right_thickness, right_thickness, down_right_thickness };

    for (const auto& se : structuring_elements) {
        for (int i = 0; i < iterations; i++) {
            Mat detected_border = hit_or_miss(output_image, se, 255);
            Mat deleted_border = complement_image(detected_border);
            bitwise_and(output_image, deleted_border, output_image);
            // imshow("Original", detected_border);
            // imshow("Result", output_image);
            // waitKey(0);
        }
    }

    return output_image;
}

/**
 * @brief Thicken the foreground objects using HMT transform
 *
 * @param input_image the binary image (255 foreground)
 * @param iterations number of iterations of thickening on each side
 * @return Mat
 */
Mat object_thickening(Mat input_image, int iterations)
{
    Mat output_image = input_image.clone();
    // Create the structuring elements for thinning/thickening
    Mat down_thickness = Mat::zeros(3, 3, CV_32SC1);
    down_thickness.at<int>(0, 0) = 255;
    down_thickness.at<int>(0, 1) = 255;
    down_thickness.at<int>(0, 2) = 255;
    down_thickness.at<int>(1, 0) = -1;
    down_thickness.at<int>(1, 2) = -1;
    Mat down_left_thickness = Mat::zeros(3, 3, CV_32SC1);
    down_left_thickness.at<int>(1, 2) = 255;
    down_left_thickness.at<int>(0, 1) = 255;
    down_left_thickness.at<int>(0, 2) = 255;
    down_left_thickness.at<int>(0, 0) = -1;
    down_left_thickness.at<int>(2, 2) = -1;
    Mat left_thickness = Mat::zeros(3, 3, CV_32SC1);
    left_thickness.at<int>(2, 0) = 255;
    left_thickness.at<int>(2, 1) = 255;
    left_thickness.at<int>(2, 2) = 255;
    left_thickness.at<int>(0, 1) = -1;
    left_thickness.at<int>(2, 1) = -1;
    Mat up_left_thickness = Mat::zeros(3, 3, CV_32SC1);
    up_left_thickness.at<int>(1, 2) = 255;
    up_left_thickness.at<int>(2, 1) = 255;
    up_left_thickness.at<int>(2, 2) = 255;
    up_left_thickness.at<int>(2, 0) = -1;
    up_left_thickness.at<int>(0, 2) = -1;
    Mat up_thickness = Mat::zeros(3, 3, CV_32SC1);
    up_thickness.at<int>(2, 0) = 255;
    up_thickness.at<int>(2, 1) = 255;
    up_thickness.at<int>(2, 2) = 255;
    up_thickness.at<int>(1, 0) = -1;
    up_thickness.at<int>(1, 2) = -1;
    Mat up_right_thickness = Mat::zeros(3, 3, CV_32SC1);
    up_right_thickness.at<int>(1, 0) = 255;
    up_right_thickness.at<int>(2, 0) = 255;
    up_right_thickness.at<int>(2, 1) = 255;
    up_right_thickness.at<int>(0, 0) = -1;
    up_right_thickness.at<int>(2, 2) = -1;
    Mat right_thickness = Mat::zeros(3, 3, CV_32SC1);
    right_thickness.at<int>(0, 0) = 255;
    right_thickness.at<int>(1, 0) = 255;
    right_thickness.at<int>(2, 0) = 255;
    right_thickness.at<int>(0, 1) = -1;
    right_thickness.at<int>(2, 1) = -1;
    Mat down_right_thickness = Mat::zeros(3, 3, CV_32SC1);
    down_right_thickness.at<int>(0, 0) = 255;
    down_right_thickness.at<int>(0, 1) = 255;
    down_right_thickness.at<int>(1, 0) = 255;
    down_right_thickness.at<int>(0, 2) = -1;
    down_right_thickness.at<int>(2, 0) = -1;
    std::vector<Mat> structuring_elements { down_thickness, down_left_thickness, left_thickness, up_left_thickness, up_thickness, up_right_thickness, right_thickness, down_right_thickness };

    for (const auto& se : structuring_elements) {
        for (int i = 0; i < iterations; i++) {
            Mat detected_border = hit_or_miss(output_image, se, 255);
            bitwise_or(output_image, detected_border, output_image);
            // imshow("Original", detected_border);
            // imshow("Result", output_image);
            // waitKey(0);
        }
    }

    return output_image;
}

/**
 * @brief Perform a geodesic dilation until convergence.
 *
 * @param marker The starting image with the point in the object
 * @param mask The desired minimum object shape
 * @param structuring_element  The structuring element
 * @return Mat
 */
Mat geodesic_dilation(Mat marker, Mat mask, Mat structuring_element)
{
    Mat last_image = marker.clone();
    Mat output_image = dilation(last_image, structuring_element, 255);
    bitwise_and(output_image, mask, output_image);
    while (sum(output_image - last_image)[0] != 0) {
        last_image = output_image.clone();
        output_image = dilation(last_image, structuring_element, 255);
        bitwise_and(output_image, mask, output_image);
    }

    return output_image;
}

/**
 * @brief Perform a geodesic erosion until convergence.
 *
 * @param marker The starting image with the full object
 * @param mask  The minimum mask of the erosion
 * @param structuring_element Used SE
 * @return Mat
 */
Mat geodesic_erosion(Mat marker, Mat mask, Mat structuring_element)
{
    Mat last_image = marker.clone();
    Mat output_image = erosion(last_image, structuring_element, 255);
    bitwise_or(output_image, mask, output_image);
    while (sum(output_image - last_image)[0] != 0) {
        last_image = output_image.clone();
        output_image = erosion(last_image, structuring_element, 255);
        bitwise_or(output_image, mask, output_image);
    }

    return output_image;
}

/**
 * @brief Reconstruct an image by opening
 *
 * @param input_image Binary image with the desired objects (255 foreground)
 * @param erosion_se The SE of erosion denoting the objects of interest
 * @param dilation_se Dilation SE for the reconstruction using the connecting elements
 * @param erosion_iterations Number of erosion iterations
 * @return Mat
 */
Mat reconstruction_by_opening(Mat input_image, Mat erosion_se, Mat dilation_se, int erosion_iterations)
{
    Mat erosion_image = input_image.clone();
    // erode the initial image to obtain the mask for reconstruction
    for (int i = 0; i < erosion_iterations; i++) {
        erosion_image = erosion(erosion_image, erosion_se, 255);
    }
    // dilate the mask and compare it to the original image
    Mat output_image = geodesic_dilation(erosion_image, input_image, dilation_se);

    return output_image;
}

Mat hole_filling_auto(Mat input_image)
{
    // SE
    Mat structuring_element = Mat(Size(3, 3), CV_32SC1, 255);
    // Create the border of complement
    Mat mask = Mat::zeros(input_image.size(), CV_8UC1);
    for (int i = 0; i < input_image.rows; i++) {
        for (int j = 0; j < input_image.cols; j++) {
            if (i == 0 || j == 0 || i == input_image.rows - 1 || j == input_image.cols - 1) {
                mask.at<uchar>(i, j) = 255 - input_image.at<uchar>(i, j);
            }
        }
    }
    // Reconstruct the outside of the borders
    Mat image_complement = complement_image(input_image);
    Mat output_image = geodesic_dilation(mask, image_complement, structuring_element);

    return complement_image(output_image);
}

/**
 * @brief Extract the partial objects in the image (which connect to the border)
 *
 * @param input_image the binary image
 * @return Mat the
 */
Mat border_objects(Mat input_image)
{
    // SE
    Mat structuring_element = Mat(Size(3, 3), CV_32SC1, 255);
    // Create the border of complement
    Mat mask = Mat::zeros(input_image.size(), CV_8UC1);
    for (int i = 0; i < input_image.rows; i++) {
        for (int j = 0; j < input_image.cols; j++) {
            if (i == 0 || j == 0 || i == input_image.rows - 1 || j == input_image.cols - 1) {
                mask.at<uchar>(i, j) = input_image.at<uchar>(i, j);
            }
        }
    }

    Mat output_image = geodesic_dilation(mask, input_image, structuring_element);

    return output_image;
}

int main(int argc, char** argv)
{
    Mat image = imread("../resources/triangle_.png", -1);
    Mat grayscale;
    cvtColor(image, grayscale, COLOR_BGR2GRAY);
    Mat gaussian;
    GaussianBlur(grayscale, gaussian, Size(5, 5), 0, 0);
    Mat threshold_image = gaussianThresholding(gaussian, 5, 2, 1);
    // Mat threshold_image = thresholdValue(gaussian, 225);
    // Mat marker = Mat::zeros(threshold_image.size(), CV_8UC1);
    // Mat structuring_element = Mat(Size(5, 105), CV_32SC1, 255);
    Mat structuring_element = Mat(Size(3, 3), CV_32SC1, 255);
    Mat hole_filled = hole_filling_auto(threshold_image);
    // Mat erosion_img = erosion(threshold_image, structuring_element, 255);

    // Mat dilation_result = dilation(threshold_image, structuring_element, 255);
    // Mat erosion_result = erosion(threshold_image, structuring_element, 255);
    // Mat border_result = border_extract(threshold_image);
    // Mat hole_fill_result = hole_filling(threshold_image, Point2d(270, 270));
    // Mat convexhull = convex_hull(threshold_image);
    // Mat thinned_img = object_thining(threshold_image);
    // Mat thickened_img = object_thickening(threshold_image, 2);
    // Mat geodesic = geodesic_dilation(marker, threshold_image, structuring_element);
    // Mat reconstruction = reconstruction_by_opening(threshold_image, structuring_element, structuring_element2, 1);
    imshow("Original", threshold_image);
    imshow("Result", hole_filled);
    waitKey(0);
    return 0;
}