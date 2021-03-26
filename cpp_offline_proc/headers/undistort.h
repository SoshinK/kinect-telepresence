//
// Created by vakhitov on 03.12.19.
//

#ifndef AZURE_KINECT_SAMPLES_UNDISTORT_H
#define AZURE_KINECT_SAMPLES_UNDISTORT_H

#include <stdio.h>
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <k4a/k4a.h>

using namespace std;

// Enable HAVE_OPENCV macro after you installed opencv and opencv contrib modules (kinfu, viz), please refer to README.md
// #define HAVE_OPENCV
//#ifdef HAVE_OPENCV
#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/rgbd.hpp>
using namespace cv;
//#endif


#define INVALID INT32_MIN
typedef struct _pinhole_t
{
    float px;
    float py;
    float fx;
    float fy;

    int width;
    int height;
} pinhole_t;

typedef struct _coordinate_t
{
    int x;
    int y;
    float weight[4];
} coordinate_t;

typedef enum
{
    INTERPOLATION_NEARESTNEIGHBOR, /**< Nearest neighbor interpolation */
    INTERPOLATION_BILINEAR,        /**< Bilinear interpolation */
    INTERPOLATION_BILINEAR_DEPTH   /**< Bilinear interpolation with invalidation when neighbor contain invalid
                                                data with value 0 */
} interpolation_t;

void create_undistortion_lut(const k4a_calibration_t* calibration,
                                    const k4a_calibration_type_t camera,
                                    const pinhole_t* pinhole,
                                    k4a_image_t lut,
                                    interpolation_t type);
template<typename T>
void remap(const k4a_image_t src, const k4a_image_t lut, k4a_image_t dst, interpolation_t type);

template <typename T>
void remap2(const k4a_image_t depth_src, const k4a_image_t src, const k4a_image_t lut, k4a_image_t dst, const size_t channels, interpolation_t type);


pinhole_t create_pinhole_from_xy_range(const k4a_calibration_t* calibration, const k4a_calibration_type_t camera);

void compute_xy_range(const k4a_calibration_t* calibration,
                             const k4a_calibration_type_t camera,
                             const int width,
                             const int height,
                             float& x_min,
                             float& x_max,
                             float& y_min,
                             float& y_max);

template<typename T> cv::Mat create_mat_from_buffer(T *data, int width, int height, int channels = 1);

#endif //AZURE_KINECT_SAMPLES_UNDISTORT_H
