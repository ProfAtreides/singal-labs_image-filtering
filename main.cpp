#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/mat.hpp>
#include <vector>
#include <algorithm>
#include <fstream>

using namespace std;
using namespace cv;

double distance(int x, int y, int neighbour_x, int neighbour_y) {
    return double(sqrt(pow(x - neighbour_x, 2) + pow(y - neighbour_y, 2)));
}

double gaussian(double x, double sigma) {
    return exp(-(pow(x, 2)) / (2 * pow(sigma, 2))) / (2 * CV_PI * pow(sigma, 2));
}

Mat kernel_convolution(Mat image,const vector <double>& kernel)
{
    int size = sqrt(kernel.size());

    Mat copy(image.rows, image.cols, CV_8UC3);

    for (int i = 0; i < image.rows; i++)
    {
        for (int j = 0; j < image.cols; j++)
        {
            int start_x = (j - size / 2 >= 0) ? 0 : j + size / 2;
            int start_y = (i - size / 2 >= 0) ? 0 : i + size / 2;
            int end_x = (j + size / 2 < image.cols) ? size : size / 2;
            int end_y = (i + size / 2 < image.rows) ? size : size / 2;

            vector <double> red_values = kernel;
            vector <double> green_values = kernel;
            vector <double> blue_values = kernel;

            double previous_sum = 0;
            double red_convoluted_sum = 0;
            double green_convoluted_sum = 0;
            double blue_convoluted_sum = 0;

            for (int y = start_y; y < end_y; y++)
            {
                for (int x = start_x; x < end_x; x++)
                {
                    Vec3b pixel = image.at<Vec3b>(i + y - size / 2, j + x - size / 2);
                    previous_sum += red_values[x + y * size];

                    blue_values[x + y * size] *=(double)(pixel[0]);
                    green_values[x + y * size] *= (double)(pixel[1]);
                    red_values[x + y * size] *= (double)(pixel[2]);

                    blue_convoluted_sum += blue_values[x + y * size];
                    green_convoluted_sum += green_values[x + y * size];
                    red_convoluted_sum += red_values[x + y * size];
                }
            }

            double temp_blue = (previous_sum > 0) ? blue_convoluted_sum / previous_sum : blue_convoluted_sum;
            double temp_green = (previous_sum > 0) ? green_convoluted_sum / previous_sum : green_convoluted_sum;
            double temp_red = (previous_sum > 0) ? red_convoluted_sum / previous_sum : red_convoluted_sum;

            if (temp_blue > 255)
            {
                temp_blue = 255;
            }
            else if (temp_blue < 0)
            {
                temp_blue = 0;
            }
            if (temp_green > 255)
            {
                temp_green = 255;
            }
            else if (temp_green < 0)
            {
                temp_green = 0;
            }
            if (temp_red > 255)
            {
                temp_red = 255;
            }
            else if (temp_red < 0)
            {
                temp_red = 0;
            }

            Vec<uchar, 3> filtered_pixel;
            filtered_pixel[0] = (uchar)(temp_blue);
            filtered_pixel[1] = (uchar)(temp_green);
            filtered_pixel[2] = (uchar)(temp_red);
            copy.at<Vec3b>(i, j) = filtered_pixel;
        }
    }

    return copy;
}

Mat median_filter(Mat image, int kernel_size)
{
    Mat copy(image.rows, image.cols, CV_8UC3);
    for (int i = 0; i < image.rows; i++)
    {
        for (int j = 0; j < image.cols; j++)
        {
            int start_x = (j - kernel_size / 2 >= 0) ? 0 : j + kernel_size / 2;
            int start_y = (i - kernel_size / 2 >= 0) ? 0 : i + kernel_size / 2;
            int end_x = (j + kernel_size / 2 < image.cols) ? kernel_size : kernel_size / 2;
            int end_y = (i + kernel_size / 2 < image.rows) ? kernel_size : kernel_size / 2;

            vector <int> red_values;
            vector <int> green_values;
            vector <int> blue_values;

            for (int y = start_y; y < end_y; y++)
            {
                for (int x = start_x; x < end_x; x++)
                {
                    Vec3b pixel = image.at<Vec3b>(i + y - kernel_size / 2, j + x - kernel_size / 2);

                    green_values.push_back((pixel[0]));
                    blue_values.push_back((pixel[1]));
                    red_values.push_back((pixel[2]));
                }
            }

            sort(green_values.begin(), green_values.end());
            sort(blue_values.begin(), blue_values.end());
            sort(red_values.begin(), red_values.end());

            int median_index = red_values.size() / 2;

            Vec<uchar, 3> filtered_pixel;
            filtered_pixel[0] = static_cast<uchar>(green_values[median_index]);
            filtered_pixel[1] = static_cast<uchar>(blue_values[median_index]);
            filtered_pixel[2] = static_cast<uchar>(red_values[median_index]);
            copy.at<Vec3b>(i, j) = filtered_pixel;
        }
    }
    return copy;
}

Mat billateral_filter(Mat image, int diameter, double sigma_color, double sigma_edge)
{
    Mat filtered(image.rows, image.cols, CV_8UC3);

    for (int i = 0; i < image.rows; i++)
    {
        for (int j = 0; j < image.cols; j++)
        {
            int start_x = (j - diameter / 2 >= 0) ? 0 : j + diameter / 2;
            int start_y = (i - diameter / 2 >= 0) ? 0 : i + diameter / 2;
            int end_x = (j + diameter / 2 < image.cols) ? diameter : diameter / 2;
            int end_y = (i + diameter / 2 < image.rows) ? diameter : diameter / 2;

            double blue_weight = 0;
            double green_weight = 0;
            double red_weight = 0;

            double blue_sum = 0;
            double green_sum = 0;
            double red_sum = 0;

            int center = diameter / 2;

            for (int y = start_y; y < end_y; y++)
            {
                for (int x = start_x; x < end_x; x++)
                {
                    int neighbour_x = j - (center - x);
                    int neighbour_y = i - (center - y);

                    if (neighbour_x < 0 || neighbour_y < 0)
                    {
                        cerr << "XD" << "\n";
                    }

                    Vec3b pixel = image.at<Vec3b>(i, j);
                    Vec3b neighbour_pixel = image.at<Vec3b>(neighbour_y, neighbour_x);

                    double gauss_distance = gaussian(distance(j, i, neighbour_x, neighbour_y), sigma_edge);

                    double gauss_blue = gaussian(static_cast<double>(neighbour_pixel[0]) - static_cast<double>(pixel[0]), sigma_color);
                    double gauss_green = gaussian(static_cast<double>(neighbour_pixel[1]) - static_cast<double>(pixel[1]), sigma_color);
                    double gauss_red = gaussian(static_cast<double>(neighbour_pixel[2]) - static_cast<double>(pixel[2]), sigma_color);

                    // Weight in current pixel
                    double temp_weight_blue = gauss_blue * gauss_distance;
                    double temp_weight_green = gauss_green * gauss_distance;
                    double temp_weight_red = gauss_red * gauss_distance;

                    blue_sum += static_cast<double>(neighbour_pixel[0]) * temp_weight_blue;
                    green_sum += static_cast<double>(neighbour_pixel[1]) * temp_weight_green;
                    red_sum += static_cast<double>(neighbour_pixel[2]) * temp_weight_red;

                    blue_weight += temp_weight_blue;
                    green_weight += temp_weight_green;
                    red_weight += temp_weight_red;
                }
            }
            if (blue_sum == 0)
            {
                cerr << "GOTCHA";
            }
            filtered.at<Vec3b>(i, j)[0] = static_cast<uchar>(blue_sum / blue_weight);
            filtered.at<Vec3b>(i, j)[1] = static_cast<uchar>(green_sum / green_weight);
            filtered.at<Vec3b>(i, j)[2] = static_cast<uchar>(red_sum / red_weight);
        }
    }
    return filtered;
}

Mat image_diff(Mat image1, Mat image2)
{
    Mat diff(image1.rows, image1.cols, CV_8UC3);

    for (int i = 0; i < image1.rows; i++)
    {
        for (int j = 0; j < image1.cols; j++)
        {
            int blue_diff = abs(image1.at<Vec3b>(i, j)[0] - image2.at<Vec3b>(i, j)[0]);
            int green_diff = abs(image1.at<Vec3b>(i, j)[1] - image2.at<Vec3b>(i, j)[1]);
            int red_diff = abs(image1.at<Vec3b>(i, j)[2] - image2.at<Vec3b>(i, j)[2]);

            diff.at<Vec3b>(i, j)[0] = blue_diff;
            diff.at<Vec3b>(i, j)[1] = green_diff;
            diff.at<Vec3b>(i, j)[2] = red_diff;
        }
    }

    return diff;
}

vector < double > avg_square_diff(Mat image, Mat filtered)
{
    int n = image.cols * image.rows;

    double avg_blue_sum = 0;
    double avg_green_sum = 0;
    double avg_red_sum = 0;

    for (int i = 0; i < n; i++)
    {
        Vec3b original_pixel = image.at<Vec3b>(i / image.rows, i % image.cols);
        Vec3b filtered_pixel = filtered.at<Vec3b>(i / image.rows, i % image.cols);

        avg_blue_sum += (static_cast<double>(original_pixel[0]) - static_cast<double>(filtered_pixel[0])) * (static_cast<double>(original_pixel[0]) - static_cast<double>(filtered_pixel[0]));
        avg_green_sum += (static_cast<double>(original_pixel[1]) - static_cast<double>(filtered_pixel[1])) * (static_cast<double>(original_pixel[1]) - static_cast<double>(filtered_pixel[1]));
        avg_red_sum += (static_cast<double>(original_pixel[2]) - static_cast<double>(filtered_pixel[2])) * (static_cast<double>(original_pixel[2]) - static_cast<double>(filtered_pixel[2]));
    }

    avg_blue_sum /= n;
    avg_green_sum /= n;
    avg_red_sum /= n;

    return { avg_blue_sum, avg_green_sum, avg_red_sum };
}

int main()
{
    ofstream log("square_diff_sth.txt");

    Mat image = imread("noise.jpg", IMREAD_COLOR);

    if (image.empty()) return 1;

    vector <double> sharp_kernel =
            { 0,-1,0,
              0,5,0,
              0,-1,0 };
    vector <double> gaussian_blur =
            { 1 ,4 ,6 ,4 ,1 ,
              4 ,16 ,24 ,16 ,4 ,
              6 ,24 ,36 ,24 ,6 ,
              4 ,16 ,24,16,4 ,
              1 ,4 ,6 ,4 ,1 };

    if (image.empty())
    {
        cerr << "ERROR" << "\n";
        return -1;
    }

    const int WIDTH = image.cols;
    const int HEIGHT = image.rows;

    Mat convolution = kernel_convolution(image, gaussian_blur);
    imwrite("test.jpg", convolution);
    imwrite("conv_diff.jpg", image_diff(image, convolution));

    Mat test_bil = billateral_filter(image, 50, 300, 250);
    imwrite("bilateral.jpg", test_bil);
    imwrite("diff_bill.jpg", image_diff(image, test_bil));

    Mat test_med = median_filter(image, 55);
    imwrite("diff_mid.jpg", image_diff(image, test_med));
    imwrite("median.jpg", test_med);

    vector <double >  convolution_diff = avg_square_diff(image, convolution);

    log << "Convolution diff: (rgb)" << "\n";
    log << convolution_diff[2] << " " << convolution_diff[1] << " " << convolution_diff[0] << "\n" << "\n";

    vector <double >  bilateral_diff = avg_square_diff(image, test_bil);

    log << "Bilateral diff: (rgb)" << "\n";
    log << bilateral_diff[2] << " " << bilateral_diff[1] << " " << bilateral_diff[0] << "\n" << "\n";

    vector <double >  median_diff = avg_square_diff(image, test_med);

    log << "Median diff: (rgb)" << "\n";
    log << median_diff[2] << " " << median_diff[1] << " " << median_diff[0] << "\n" << "\n";

    log.close();
}