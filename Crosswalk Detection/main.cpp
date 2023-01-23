#include <filesystem>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <string>

// for brevity
namespace fs = std::filesystem;
using namespace cv;

/**
 * @brief Detect the crosswalk based on its geometric properties.
 *
 * @param input_image input image
 * @param minArea Minimum area required for a contour to be valid.
 * @param maxPolygonSides Max accepted sides for the polygon approximation.
 * @param minMatches Minimum contour matches required to accept it as a crosswalk.
 * @return Mat
 */
Mat detectCrosswalk(Mat input_image, int minArea, int maxPolygonSides, int minMatches)
{
    // Considered color filtering but it is not really helpful
    // This is the result that different lighting can have on the crosswalk + different crosswalks color
    // Resize the image
    Mat resized_image;
    resize(input_image, resized_image, Size(848, 480), INTER_LINEAR);
    // Grayscale the image
    Mat grayscale_image;
    cvtColor(resized_image, grayscale_image, COLOR_BGR2GRAY);

    // Contrast strech to use all the domain of the image
    Mat equalized_image;
    equalizeHist(grayscale_image, equalized_image);
    // Blur the image
    Mat gaussian_image;
    GaussianBlur(grayscale_image, gaussian_image, Size(5, 5), 0);

    // Threshold given high intensity of the crosswalk
    Mat threshold_image;
    adaptiveThreshold(gaussian_image, threshold_image, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 3, 2);
    // // // Morpholiogical operations on the binary image.
    Mat morph_image;
    int morph_size = 2;
    Mat element = getStructuringElement(MORPH_RECT, Size(2 * morph_size + 1, 2 * morph_size + 1), Point(morph_size, morph_size));
    morphologyEx(threshold_image, morph_image, MORPH_DILATE, element, Point(-1, -1), 1); // further enhance the contours
    // morphologyEx(morph_image, morph_image, MORPH_GRADIENT, element, Point(-1, -1), 1); // further enhance the contours
    int morph_size_2 = 1;
    Mat element2 = getStructuringElement(MORPH_RECT, Size(2 * morph_size_2 + 1, 2 * morph_size_2 + 1), Point(morph_size_2, morph_size_2));
    morphologyEx(morph_image, morph_image, MORPH_OPEN, element2, Point(-1, -1), 1); // further enhance the contours

    // Approximate the contours of the image
    // detect the contours on the binary image using cv2.CHAIN_APPROX_NONE
    std::vector<std::vector<Point>> contours;
    std::vector<Vec4i> hierarchy;
    findContours(morph_image, contours, hierarchy, RETR_TREE, CHAIN_APPROX_NONE);

    Mat contour_image = Mat::zeros(threshold_image.rows, threshold_image.cols, CV_8U);
    std::vector<std::vector<Point>> good_contours;
    // append the relevant contours
    for (int i = 0; i < contours.size(); i++) {
        double area = cv::contourArea(contours[i]);
        if (area > minArea) { // parameters of the function
            // Given the arc length of the contour, take an error (epsilon)
            double epsilon = 0.01 * arcLength(contours[i], true);
            std::vector<Point> contour_approximation;
            approxPolyDP(contours[i], contour_approximation, epsilon, true);
            if (contour_approximation.size() > 3 && contour_approximation.size() < maxPolygonSides) { // max octagon
                // append the good contours
                good_contours.push_back(contours[i]);
            }
        }
    }
    // draw the contours
    // get the angles orientation of contours
    std::vector<Point2f> angle_contours;
    for (size_t i = 0; i < good_contours.size(); i++) {
        // fit an elipse to the desired object
        RotatedRect approximation = fitEllipse(good_contours[i]);
        angle_contours.push_back(Point2f(approximation.angle, i));
    }

    std::vector<int> crosswalk_indexes;
    // get similar angles
    // match between each contour slope appproximation and add only those with a minimum pair of 3 contours
    for (int i = 0; i < good_contours.size(); i++) {
        int matches = 0;
        for (int j = 0; j < good_contours.size(); j++) {
            if (abs((abs(angle_contours[i].x) - abs(angle_contours[j].x))) < 5) {
                matches++;
            }
        }
        if (matches >= minMatches) { // if we have enough matches in the noise or we barely have a contour to match with
            crosswalk_indexes.push_back(angle_contours[i].y);
        }
    }

    // draw the resulting contours in a binary image
    std::vector<std::vector<Point>> hulls(crosswalk_indexes.size());

    // find the total bounding rectangle
    std::vector<Point> total_contour_points;
    for (size_t i = 0; i < crosswalk_indexes.size(); i++) {
        total_contour_points.insert(std::end(total_contour_points), std::begin(good_contours[crosswalk_indexes[i]]), std::end(good_contours[crosswalk_indexes[i]]));
        // OPTIONAL to draw the binary image result
        convexHull(good_contours[crosswalk_indexes[i]], hulls[i]);
        drawContours(contour_image, hulls, (int)i, 255);
    }

    Rect bounding_rect = boundingRect(total_contour_points);
    Mat output_image = resized_image.clone();
    rectangle(output_image, bounding_rect.tl(), bounding_rect.br(), Scalar(0, 255, 0), 2);
    // draw results
    // imshow("Original", morph_image);
    // imshow("Output image", output_image);
    // waitKey(0);
    return output_image;
}

int main(int argc, char** argv)
{
    std::string path = "../dataset/";
    int idx = 0;
    for (const auto& entry : fs::directory_iterator(path)) {
        Mat input_img = imread(entry.path(), -1);
        Mat output_img = detectCrosswalk(input_img, 35 * 35, 8, 3);
        std::string write_path = "../results/result" + std::to_string(idx) + ".jpg";
        imwrite(write_path, output_img);
        idx += 1;
    }
    // Mat input_img = imread("../dataset/crosswalk6.jpg", -1);
    // Mat output_img = detectCrosswalk(input_img, 35, 8);
    return 0;
}