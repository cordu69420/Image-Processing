#include <iostream>
#include <opencv2/opencv.hpp>
using namespace cv;

/**
 * @brief scale the given image using the @param scale_factor.
 * 
 * @param input_image : The input image provided for the scale process.
 * @param scale_factor : The float value which gives scale property.
 * @return Mat 
 */
Mat scaleImage(Mat input_image, float scale_factor){
    
    // check for valid input
    if(!input_image.data){
        throw "Bad input.";
    }

    // obtain the original dimensions 
    int rows = input_image.rows;
    int cols = input_image.cols;
    
    // scale the columns and rows
    int rows_scaled = std::round(rows * scale_factor);
    int cols_scaled = std::round(cols * scale_factor);

    Mat output_image(cols_scaled, rows_scaled, CV_8UC3);

    for(int i = 0 ; i < rows_scaled; i++){
        // calculate the new i index
        int x_index = std::round(i / scale_factor);
        for(int j = 0; j < cols_scaled; j++){
            // calculate the new j index
            int y_index = std::round(j / scale_factor);
            output_image.at<Vec3b>(i,j) = input_image.at<Vec3b>(x_index, y_index);
        }
    }

    return output_image;
}

/**
 * @brief Interpolate the correct scaling value given a bilinear alghorithm
 * 
 * @param input_image The input image provided for the scale process.
 * @param x_index The current y index in the new image
 * @param y_index The current x index in the new image
 * @return Vec3b color result
 */
Vec3b interpolateValue(Mat input_image, int x_index, int y_index){
    
    // A * X = P
    // Define the neighbour matrix A
    float x_index_float = (float)x_index;
    float y_index_float = (float)y_index;
    float neighbour_array[4][4] = {
        {x_index_float - 1, y_index_float - 1, (x_index_float - 1)*(y_index_float - 1), 1},
        {x_index_float - 1, y_index_float + 1, (x_index_float - 1)*(y_index_float + 1), 1},
        {x_index_float + 1, y_index_float - 1, (x_index_float + 1)*(y_index_float - 1), 1},
        {x_index_float + 1, y_index_float + 1, (x_index_float + 1)*(y_index_float + 1), 1},
    };
    Mat neighbour_mat = Mat(4, 4, CV_32FC1, &neighbour_array);

    // declare the results of the equation P
    float results_r[1][4] = {{input_image.at<Vec3b>(x_index_float - 1, y_index_float - 1)[0], input_image.at<Vec3b>(x_index_float - 1, y_index_float + 1)[0], input_image.at<Vec3b>(x_index_float + 1, y_index_float - 1)[0], input_image.at<Vec3b>(x_index_float + 1, y_index_float + 1)[0]}};
    float results_g[1][4] = {{input_image.at<Vec3b>(x_index_float - 1, y_index_float - 1)[1], input_image.at<Vec3b>(x_index_float - 1, y_index_float + 1)[1], input_image.at<Vec3b>(x_index_float + 1, y_index_float - 1)[1], input_image.at<Vec3b>(x_index_float + 1, y_index_float + 1)[1]}};
    float results_b[1][4] = {{input_image.at<Vec3b>(x_index_float - 1, y_index_float - 1)[2], input_image.at<Vec3b>(x_index_float - 1, y_index_float + 1)[2], input_image.at<Vec3b>(x_index_float + 1, y_index_float - 1)[2], input_image.at<Vec3b>(x_index_float + 1, y_index_float + 1)[2]}};
    
    Mat results_mat_r = Mat(4, 1, CV_32FC1, &results_r);
    Mat results_mat_g = Mat(4, 1, CV_32FC1, &results_g);
    Mat results_mat_b = Mat(4, 1, CV_32FC1, &results_b);

    Mat neighbour_mat_inv = neighbour_mat.inv();
    // X = A^-1 * P
    Mat solution_r = neighbour_mat_inv * results_mat_r;
    Mat solution_g = neighbour_mat_inv * results_mat_g;
    Mat solution_b = neighbour_mat_inv * results_mat_b;
    
    float r_result = solution_r.at<float>(0,0)*x_index_float + solution_r.at<float>(0,1)*y_index_float + solution_r.at<float>(0,2)*x_index_float*y_index_float + solution_r.at<float>(0,3);
    float g_result = solution_g.at<float>(0,0)*x_index_float + solution_g.at<float>(0,1)*y_index_float + solution_g.at<float>(0,2)*x_index_float*y_index_float + solution_g.at<float>(0,3);
    float b_result = solution_b.at<float>(0,0)*x_index_float + solution_b.at<float>(0,1)*y_index_float + solution_b.at<float>(0,2)*x_index_float*y_index_float + solution_b.at<float>(0,3);

    // v(x, y) = a*x +b*y + c*x*y + d
    Vec3b pixel_value = Vec3b(r_result, g_result, b_result);
    
    return pixel_value;
}

/**
 * @brief scale the given image using the bilinear scalling algorithm
 * 
 * @param input_image : The input image provided for the scale process.
 * @param scale_factor : The float value which gresultives scale property.
 * @return Mat 
 */
Mat bilinearScalling(Mat input_image, float scale_factor){
     
    // check for valid input
    if(!input_image.data){
        throw "Bad input.";
    }

    // obtain the original dimensions 
    int rows = input_image.rows;
    int cols = input_image.cols;
    
    // scale the columns and rows
    int rows_scaled = std::round(rows * scale_factor);
    int cols_scaled = std::round(cols * scale_factor);

    Mat output_image(cols_scaled, rows_scaled, CV_8UC3);

    for(int i = 0 ; i < rows_scaled; i++){
        // calculate the new i index
        int x_index = std::round(i / scale_factor);
        for(int j = 0; j < cols_scaled; j++){
            // calculate the new j index
            int y_index = std::round(j / scale_factor);
            Vec3b result;
            if(x_index > 0 && y_index > 0 && x_index < rows && y_index < cols){
                result = interpolateValue(input_image, x_index, y_index);
            }else{
                result = input_image.at<Vec3b>(x_index, y_index);
            }
            output_image.at<Vec3b>(i,j) = result;
        }
    }

    return output_image;
}

int main(int argc, char** argv )
{

    Mat input_image;
    input_image = imread( "../resources/lena.png", -1);
    
    Mat result_linear = scaleImage(input_image, 0.5);
    Mat result_bilinear = bilinearScalling(input_image, 0.5);

    imwrite("../resources/result_linear.png", result_linear);
    imwrite("../resources/result_bilinear.png", result_bilinear);
    
    return 0;
}