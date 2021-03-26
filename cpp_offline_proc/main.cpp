// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <iostream>
#include <fstream>
#include <string>
#include <iomanip>
#include <chrono>
#include "opencv2/core.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include <k4a/k4a.h>
#include <k4arecord/playback.h>
#include <k4abt.h>
#include <nlohmann/json.hpp>

//#include <BodyTrackingHelpers.h>
//#include <Utilities.h>

#include "undistort.h"
#include "ThreadPool/ThreadPool.h"

using namespace std::chrono;
using namespace nlohmann;
using namespace cv;

/*bool get_frame_bt_json(k4abt_frame_t& body_frame, int frame_num, json& frame_bt_json) {
    size_t num_bodies = k4abt_frame_get_num_bodies(body_frame);
    frame_bt_json["timestamp_usec"] = k4abt_frame_get_device_timestamp_usec(body_frame);
    frame_bt_json["frame_num"] = frame_num;
    frame_bt_json["num_bodies"] = num_bodies;
    frame_bt_json["bodies"] = json::array();
    for (size_t i = 0; i < num_bodies; i++) {
        k4abt_skeleton_t skeleton;
        VERIFY(k4abt_frame_get_body_skeleton(body_frame, i, &skeleton), "Get body from body frame failed!");
        json body_json;
        int body_id = k4abt_frame_get_body_id(body_frame, i);
        body_json["body_id"] = body_id;

        for (int j = 0; j < (int)K4ABT_JOINT_COUNT; j++) {
            body_json["joint_positions"].push_back({
                skeleton.joints[j].position.xyz.x,
                skeleton.joints[j].position.xyz.y,
                skeleton.joints[j].position.xyz.z
            });
            body_json["joint_orientations"].push_back({
                skeleton.joints[j].orientation.wxyz.w,
                skeleton.joints[j].orientation.wxyz.x,
                skeleton.joints[j].orientation.wxyz.y,
                skeleton.joints[j].orientation.wxyz.z
            });
            body_json["joint_confidences"].push_back(int(skeleton.joints[j].confidence_level));

        }
        frame_bt_json["bodies"].push_back(body_json);
    }
    return true;
}
*/
bool execute_bt(
    json& frames_json, 
    int frame_num,
    k4abt_tracker_t& tracker, 
    k4a_capture_t& capture_handle,
    k4a_image_t& lut,
    bool save_bt_json,
    bool save_bt_index_map, bool save_bt_index_map_undistorted,
    UMat& index_map_umat, UMat& index_map_umat_undistorted
) {
    k4a_wait_result_t queue_capture_result = k4abt_tracker_enqueue_capture(tracker, capture_handle, K4A_WAIT_INFINITE);
    if (queue_capture_result != K4A_WAIT_RESULT_SUCCEEDED) {
        std::cerr << "Error! Adding capture to tracker process queue failed!" << std::endl;
        return false;
    }

    k4abt_frame_t body_frame = nullptr;
    k4a_wait_result_t pop_frame_result = k4abt_tracker_pop_result(tracker, &body_frame, K4A_WAIT_INFINITE);
    if (pop_frame_result != K4A_WAIT_RESULT_SUCCEEDED) {
        std::cerr << "Error! Popping body tracking result failed!" << std::endl;
        return false;
    }
    if (save_bt_json) {
        /*json frame_bt_json;
        if (get_frame_bt_json(body_frame, frame_num, frame_bt_json)) {
            frames_json.push_back(frame_bt_json);
        } else {
            return false;
        }*/
    }

    if (save_bt_index_map || save_bt_index_map_undistorted) {
        k4a_image_t index_map = k4abt_frame_get_body_index_map(body_frame);
        
        int width = k4a_image_get_width_pixels(index_map);
        int height = k4a_image_get_height_pixels(index_map);

        if (save_bt_index_map_undistorted) {
            k4a_image_t index_map_undistorted = NULL;
            k4a_image_create(
                K4A_IMAGE_FORMAT_CUSTOM8,
                width,
                height,
                width * (int)sizeof(uint8_t),
                &index_map_undistorted
            );
            remap<uint8_t>(index_map, lut, index_map_undistorted, INTERPOLATION_NEARESTNEIGHBOR);
            uint8_t* buffer = k4a_image_get_buffer(index_map_undistorted);
            create_mat_from_buffer<uint8_t>(buffer, width, height).copyTo(index_map_umat_undistorted);
            k4a_image_release(index_map_undistorted);
        }
        if (save_bt_index_map) {
            uint8_t* buffer = k4a_image_get_buffer(index_map);
            create_mat_from_buffer<uint8_t>(buffer, width, height).copyTo(index_map_umat);
        }

        k4a_image_release(index_map);
    }
    k4abt_frame_release(body_frame);
    return true;
}

bool check_resolution(
    int h_calib, int h_img,
    int w_calib, int w_img,
    std::string stream
) {
    if (h_calib != h_img) {
        std::cerr << "h_calib != h_img for stream " << stream << ": " << h_calib << " != " << h_img << std::endl;
        return false;
    }
    if (w_calib != w_img) {
        std::cerr << "w_calib != w_img for stream " << stream << ": " << w_calib << " != " << w_img << std::endl;
        return false;
    }
    return true;
}

bool create_transformed_images(
    k4a_calibration_t& calibration,
    k4a_transformation_t& transformation,
    k4a_image_t& color_image, k4a_image_t& depth_image, 
    k4a_image_t* transformed_depth_image, k4a_image_t* transformed_color_image,
    k4a_image_t* depth_camera_points, k4a_image_t* color_camera_points
) {
    int h_color = k4a_image_get_height_pixels(color_image);
    int w_color = k4a_image_get_width_pixels(color_image);

    int h_depth = k4a_image_get_height_pixels(depth_image);
    int w_depth = k4a_image_get_width_pixels(depth_image);

    // transform depth image to color camera
    auto t0 = high_resolution_clock::now(); 
    if (K4A_RESULT_SUCCEEDED != k4a_image_create(
        K4A_IMAGE_FORMAT_DEPTH16,
        w_color, h_color,
        w_color * (int)sizeof(uint16_t),
        transformed_depth_image
    )) {
        std::cerr << "Failed to create transformed depth image" << std::endl;
        return false;
    }
    if (K4A_RESULT_SUCCEEDED != k4a_transformation_depth_image_to_color_camera(
        transformation, 
        depth_image, 
        *transformed_depth_image
    )) {
        std::cerr << "Failed to compute transformed depth image" << std::endl;
        return false;
    }

    // transform color image to depth camera
    auto t1 = high_resolution_clock::now(); 
    if (K4A_RESULT_SUCCEEDED != k4a_image_create(
        K4A_IMAGE_FORMAT_COLOR_BGRA32,
        w_depth, h_depth,
        w_depth * 4 * (int)sizeof(uint8_t),
        transformed_color_image
    )) {
        std::cerr << "Failed to create transformed color image" << std::endl;
        return false;
    }
    if (K4A_RESULT_SUCCEEDED != k4a_transformation_color_image_to_depth_camera(
        transformation,
        depth_image,
        color_image,
        *transformed_color_image
    )) {
        std::cerr << "Failed to compute transformed color image" << std::endl;
        return false;
    }

    // depth_camera_points
    auto t2 = high_resolution_clock::now(); 
    if (K4A_RESULT_SUCCEEDED != k4a_image_create(
        K4A_IMAGE_FORMAT_CUSTOM,
        w_depth, h_depth,
        w_depth * 3 * (int)sizeof(int16_t),
        depth_camera_points
    )) {
        std::cerr << "Failed to create point cloud image for depth camera" << std::endl;
        return false;
    }
    if (K4A_RESULT_SUCCEEDED != k4a_transformation_depth_image_to_point_cloud(
        transformation,
        depth_image,
        K4A_CALIBRATION_TYPE_DEPTH,
        *depth_camera_points
    )) {
        std::cerr << "Failed to compute point cloud for depth camera" << std::endl;
        return false;
    }

    // color_camera_points
    auto t3 = high_resolution_clock::now(); 
    k4a_image_t point_cloud_image = NULL;
    if (K4A_RESULT_SUCCEEDED != k4a_image_create(
        K4A_IMAGE_FORMAT_CUSTOM,
        w_color, h_color,
        w_color * 3 * (int)sizeof(int16_t),
        color_camera_points
    )) {
        std::cerr << "Failed to create point cloud image for color camera" << std::endl;
        return false;
    }
    if (K4A_RESULT_SUCCEEDED != k4a_transformation_depth_image_to_point_cloud(
        transformation,
        *transformed_depth_image,
        K4A_CALIBRATION_TYPE_COLOR,
        *color_camera_points
    )) {
        std::cerr << "Failed to compute point cloud for color camera" << std::endl;
        return false;
    }

    auto t4 = high_resolution_clock::now(); 
    auto create_transformed_depth_image = duration_cast<microseconds>(t1 - t0).count() / 1e6;
    auto create_transformed_color_image = duration_cast<microseconds>(t2 - t1).count() / 1e6;
    auto create_depth_camera_points = duration_cast<microseconds>(t3 - t2).count() / 1e6;
    auto create_color_camera_points = duration_cast<microseconds>(t4 - t3).count() / 1e6;
    // std::cout << "create depth_camera pc: " << create_transformed_color_image + create_depth_camera_points << std::endl;
    // std::cout << "create color_camera pc: " << create_transformed_depth_image + create_color_camera_points << std::endl;

    return true;
}

bool split_point_cloud(k4a_image_t point_cloud, uint16_t* x, uint16_t* y, uint16_t* z) {
    int width = k4a_image_get_width_pixels(point_cloud);
    int height = k4a_image_get_height_pixels(point_cloud);

    int16_t *point_cloud_data = (int16_t *)(void *)k4a_image_get_buffer(point_cloud);
    
    for (int i = 0; i < width * height; i++) {
        x[i] = point_cloud_data[3 * i + 0];
        y[i] = point_cloud_data[3 * i + 1];
        z[i] = point_cloud_data[3 * i + 2];
    }

    return true;
}

json get_intrinsics(k4a_calibration_intrinsic_parameters_t * intrinsics) {
    json result;
    result["fx"] = intrinsics->param.fx;
    result["fy"] = intrinsics->param.fy;
    result["cx"] = intrinsics->param.cx;
    result["cy"] = intrinsics->param.cy;

    result["k1"] = intrinsics->param.k1;
    result["k2"] = intrinsics->param.k2;
    result["p1"] = intrinsics->param.p1;
    result["p2"] = intrinsics->param.p2;
    result["k3"] = intrinsics->param.k3;
    result["k4"] = intrinsics->param.k4;
    result["k5"] = intrinsics->param.k5;
    result["k6"] = intrinsics->param.k6;
    return result;
} 

void output_intrinsics(
    int w_color, int h_color,
    int w_depth, int h_depth,
    k4a_calibration_intrinsic_parameters_t * depth_intrinsics,
    k4a_calibration_intrinsic_parameters_t * rgb_intrinsics, 
    Mat t_vec, Mat r_vec, 
    json& json_output
) {
    json color_resolution;
    color_resolution["w"] = w_color;
    color_resolution["h"] = h_color;
    json_output["color_resolution"] = color_resolution;

    json depth_resolution;
    depth_resolution["w"] = w_depth;
    depth_resolution["h"] = h_depth;
    json_output["depth_resolution"] = depth_resolution;

    json_output["depth_to_rgb"] = json::array();

    json depth_to_rgb_json, r_json, t_json;
    for (int i = 0; i < 3; i++) {
        r_json.push_back(r_vec.at<float>(i));
    }
    depth_to_rgb_json["r"] = r_json;
    
    for (int i = 0; i < 3; i++) {
        t_json.push_back(t_vec.at<float>(i));
    }
    depth_to_rgb_json["t"] = t_json;
    json_output["depth_to_rgb"] = depth_to_rgb_json;

    json_output["depth_intrinsics"] = get_intrinsics(depth_intrinsics);
    json_output["rgb_intrinsics"] = get_intrinsics(rgb_intrinsics);
}


void output_pinhole(const std::string& label, pinhole_t& pinhole, json& json_output) {
    json tmp;
    tmp["fx"] = pinhole.fx;
    tmp["fy"] = pinhole.fy;
    tmp["px"] = pinhole.px;
    tmp["py"] = pinhole.py;
    json_output[label] = tmp;
}


void output_cam_mat(const std::string& label, Mat& cam_mat, json& json_output) {
    json tmp;
    tmp["fx"] = cam_mat.at<float>(0,0);
    tmp["fy"] = cam_mat.at<float>(1,1);
    tmp["cx"] = cam_mat.at<float>(0,2);
    tmp["cy"] = cam_mat.at<float>(1,2);
    json_output[label] = tmp;
}

bool write_json(std::string output_filepath, json& data) {
    std::ofstream file_id;
    file_id.open(output_filepath);
    if (!file_id) {
        std::cerr << "failed to open file " << output_filepath << std::endl;
        return false;
    } else {
        file_id << std::setw(4) << data << std::endl;
        file_id.close();
        return true;
    }
}

bool save_mat(std::string output_filepath, Mat img) {
    imwrite(output_filepath, img);
}

bool save_umat(std::string output_filepath, UMat img) {
    imwrite(output_filepath, img);
}

bool process_mkv_offline(
    const char* input_path, 
    const std::string output_dirpath,
    bool save_color,
    bool save_color_undistorted,
    bool save_depth,
    bool save_depth_undistorted,
    bool save_ts_json,
    bool save_bt_json,
    bool save_bt_index_map,
    bool save_bt_index_map_undistorted,
    bool save_transformed_depth,
    bool save_transformed_depth_undistorted,
    bool save_transformed_color,
    bool save_transformed_color_undistorted,
    bool save_depth_point_cloud,
    bool save_color_point_cloud,
    int rotate,
    bool save_jpg
) {
    if (save_transformed_depth_undistorted) {
        std::cout << "save_transformed_depth_undistorted not tested properly" << std::endl;
        // return false;
    }

    json flags;
    flags["save_color"] = save_color;
    flags["save_color_undistorted"] = save_color_undistorted;
    flags["save_depth"] = save_depth;
    flags["save_depth_undistorted"] = save_depth_undistorted;
    flags["save_ts_json"] = save_ts_json;
    flags["save_bt_json"] = save_bt_json;
    flags["save_bt_index_map"] = save_bt_index_map;
    flags["save_bt_index_map_undistorted"] = save_bt_index_map_undistorted;
    flags["save_transformed_depth"] = save_transformed_depth;
    flags["save_transformed_depth_undistorted"] = save_transformed_depth_undistorted;
    flags["save_transformed_color"] = save_transformed_color;
    flags["save_transformed_color_undistorted"] = save_transformed_color_undistorted;
    flags["save_depth_point_cloud"] = save_depth_point_cloud;
    flags["save_color_point_cloud"] = save_color_point_cloud;
    flags["rotate"] = rotate;
    flags["jpg"] = save_jpg;

    bool save_bt = save_bt_json || save_bt_index_map || save_bt_index_map_undistorted;
    bool save_transformed_images = (
        save_transformed_depth || save_transformed_depth_undistorted || 
        save_transformed_color || save_transformed_color_undistorted || 
        save_depth_point_cloud || save_color_point_cloud
    );
    bool process_frames = (
        save_color || save_color_undistorted || 
        save_depth || save_depth_undistorted || 
        save_ts_json || 
        save_bt || 
        save_transformed_images
    );

    k4a_playback_t playback_handle = nullptr;
    k4a_result_t result = k4a_playback_open(input_path, &playback_handle);
    if (result != K4A_RESULT_SUCCEEDED) {
        std::cerr << "Cannot open recording at " << input_path << std::endl;
        return false;
    }

    k4a_playback_set_color_conversion(playback_handle, k4a_image_format_t::K4A_IMAGE_FORMAT_COLOR_BGRA32);

    k4a_calibration_t calibration;
    result = k4a_playback_get_calibration(playback_handle, &calibration);
    if (result != K4A_RESULT_SUCCEEDED) {
        std::cerr << "Failed to get calibration" << std::endl;
        return false;
    }
    k4a_transformation_t transformation = k4a_transformation_create(&calibration);
    float* a = calibration.depth_camera_calibration.extrinsics.rotation;
    float* b =  calibration.depth_camera_calibration.extrinsics.translation;

    cout << "HEY!! " << transformation << '\n' << "HOPA1!! " << a[0] <<' '<< a[1] << ' ' <<  a[2] << ' '<< a[3] << ' ' << a[4] << ' ' <<  a[5]<< ' ' << a[6]<< ' ' << a[7]<< ' ' << a[8] << '\n';
    cout << "lala " << b[0] << ' '<< b[1] << ' ' << b[2] << '\n';

    k4abt_tracker_t tracker = NULL;
    k4abt_tracker_configuration_t tracker_config = K4ABT_TRACKER_CONFIG_DEFAULT;
    if (rotate != 0) {
        // std::cout << tracker_config.sensor_orientation << std::endl;
        if (rotate == 1) {
            // std::cout << "K4ABT_SENSOR_ORIENTATION_COUNTERCLOCKWISE90" << std::endl;
            tracker_config.sensor_orientation = K4ABT_SENSOR_ORIENTATION_COUNTERCLOCKWISE90;
        } else if (rotate == -1) {
            // std::cout << "K4ABT_SENSOR_ORIENTATION_CLOCKWISE90" << std::endl;
            tracker_config.sensor_orientation = K4ABT_SENSOR_ORIENTATION_CLOCKWISE90;
        }
        // std::cout << tracker_config.sensor_orientation <<  std::endl;
    }
    json frames_tracking;
    if (save_bt) {
        if (K4A_RESULT_SUCCEEDED != k4abt_tracker_create(&calibration, tracker_config, &tracker)) {
            std::cerr << "Body tracker initialization failed!" << std::endl;
            return false;
        }
        frames_tracking = json::array();
    }

    int frame_num = -1;
    json frames_ts = json::array();

    std::string output_color_dirpath = output_dirpath + "/" + "color";
    std::string output_color_undistorted_dirpath = output_dirpath + "/" + "color_undistorted";
    std::string output_depth_dirpath = output_dirpath + "/" + "depth";
    std::string output_depth_undistorted_dirpath = output_dirpath + "/" + "depth_undistorted";
    std::string output_bt_index_map_dirpath = output_dirpath + "/" + "bt_index_map";
    std::string output_bt_index_map_undistorted_dirpath = output_dirpath + "/" + "bt_index_map_undistorted";
    std::string output_transformed_depth_dirpath = output_dirpath + "/" + "transformed_depth";
    std::string output_transformed_depth_undistorted_dirpath = output_dirpath + "/" + "transformed_depth_undistorted";
    std::string output_transformed_color_dirpath = output_dirpath + "/" + "transformed_color";
    std::string output_transformed_color_undistorted_dirpath = output_dirpath + "/" + "transformed_color_undistorted";
    std::string output_depth_point_cloud_dirpath_x = output_dirpath + "/" + "depth_point_cloud_x";
    std::string output_depth_point_cloud_dirpath_y = output_dirpath + "/" + "depth_point_cloud_y";
    std::string output_depth_point_cloud_dirpath_z = output_dirpath + "/" + "depth_point_cloud_z";
    std::string output_color_point_cloud_dirpath_x = output_dirpath + "/" + "color_point_cloud_x";
    std::string output_color_point_cloud_dirpath_y = output_dirpath + "/" + "color_point_cloud_y";
    std::string output_color_point_cloud_dirpath_z = output_dirpath + "/" + "color_point_cloud_z";

    system(("mkdir -p " + output_dirpath).c_str());
    if (save_color) {
        system(("mkdir -p " + output_color_dirpath).c_str());
    }
    if (save_color_undistorted) {
        system(("mkdir -p " + output_color_undistorted_dirpath).c_str());
    }
    if (save_depth) {
        system(("mkdir -p " + output_depth_dirpath).c_str());
    }
    if (save_depth_undistorted) {
        system(("mkdir -p " + output_depth_undistorted_dirpath).c_str());
    }
    if (save_bt_index_map) {
        system(("mkdir -p " + output_bt_index_map_dirpath).c_str());
    }
    if (save_bt_index_map_undistorted) {
        system(("mkdir -p " + output_bt_index_map_undistorted_dirpath).c_str());
    }
    if (save_transformed_depth) {
        system(("mkdir -p " + output_transformed_depth_dirpath).c_str());
    }
    if (save_transformed_depth_undistorted) {
        system(("mkdir -p " + output_transformed_depth_undistorted_dirpath).c_str());
    }
    if (save_transformed_color) {
        system(("mkdir -p " + output_transformed_color_dirpath).c_str());
    }
    if (save_transformed_color_undistorted) {
        system(("mkdir -p " + output_transformed_color_undistorted_dirpath).c_str());
    }
    if (save_depth_point_cloud) {
        system(("mkdir -p " + output_depth_point_cloud_dirpath_x).c_str());
        system(("mkdir -p " + output_depth_point_cloud_dirpath_y).c_str());
        system(("mkdir -p " + output_depth_point_cloud_dirpath_z).c_str());
    }
    if (save_color_point_cloud) {
        system(("mkdir -p " + output_color_point_cloud_dirpath_x).c_str());
        system(("mkdir -p " + output_color_point_cloud_dirpath_y).c_str());
        system(("mkdir -p " + output_color_point_cloud_dirpath_z).c_str());
    }

    json mkv_meta;
    mkv_meta["k4abt_sdk_version"] = K4ABT_VERSION_STR;
    mkv_meta["source_file"] = input_path;

    mkv_meta["joint_names"] = json::array();
    //for (int i = 0; i < (int)K4ABT_JOINT_COUNT; i++) {
    //    mkv_meta["joint_names"].push_back(g_jointNames.find((k4abt_joint_id_t)i)->second);
    //}

    // Store all bone linkings to the json
    mkv_meta["bone_list"] = json::array();
    /*for (int i = 0; i < (int)g_boneList.size(); i++) {
        mkv_meta["bone_list"].push_back({
            g_jointNames.find(g_boneList[i].first)->second,
            g_jointNames.find(g_boneList[i].second)->second
        });
    }*/

    int w_color = calibration.color_camera_calibration.resolution_width;
    int h_color = calibration.color_camera_calibration.resolution_height;

    int w_depth = calibration.depth_camera_calibration.resolution_width;
    int h_depth = calibration.depth_camera_calibration.resolution_height;

    Mat se3 = Mat(
        3, 3, 
        CV_32FC1, 
        calibration.extrinsics[K4A_CALIBRATION_TYPE_DEPTH][K4A_CALIBRATION_TYPE_COLOR].rotation
    );
    Mat r_vec = Mat(3, 1, CV_32FC1);
    Rodrigues(se3, r_vec);
    Mat t_vec = Mat(
        3, 1, 
        CV_32F, 
        calibration.extrinsics[K4A_CALIBRATION_TYPE_DEPTH][K4A_CALIBRATION_TYPE_COLOR].translation
    );

    k4a_calibration_intrinsic_parameters_t *depth_intrinsics = &calibration.depth_camera_calibration.intrinsics.parameters;
    k4a_calibration_intrinsic_parameters_t *rgb_intrinsics = &calibration.color_camera_calibration.intrinsics.parameters;

    output_intrinsics(
        w_color, h_color,
        w_depth, h_depth,
        depth_intrinsics, rgb_intrinsics,
        t_vec, r_vec, 
        mkv_meta
    );

    
    // rgb intrinsics
    vector<float> _camera_matrix_rgb = {
        rgb_intrinsics->param.fx, 0.f, rgb_intrinsics->param.cx, 
        0.f, rgb_intrinsics->param.fy, rgb_intrinsics->param.cy, 
        0.f, 0.f, 1.f
    };
    Mat camera_matrix_rgb = Mat(3, 3, CV_32F, &_camera_matrix_rgb[0]);
    vector<float> _dist_coeffs_rgb = { 
        rgb_intrinsics->param.k1, rgb_intrinsics->param.k2, 
        rgb_intrinsics->param.p1, rgb_intrinsics->param.p2, 
        rgb_intrinsics->param.k3, rgb_intrinsics->param.k4,
        rgb_intrinsics->param.k5, rgb_intrinsics->param.k6
    };
    Mat dist_coeffs_rgb = Mat(8, 1, CV_32F, &_dist_coeffs_rgb[0]);
    Mat new_camera_matrix_rgb = camera_matrix_rgb.clone();
    new_camera_matrix_rgb.at<float>(1,1) = new_camera_matrix_rgb.at<float>(0,0);

    // // depth intrinsics
    // vector<float> _camera_matrix_depth = {
    //     depth_intrinsics->param.fx, 0.f, depth_intrinsics->param.cx, 
    //     0.f, depth_intrinsics->param.fy, depth_intrinsics->param.cy, 
    //     0.f, 0.f, 1.f
    // };
    // Mat camera_matrix_depth = Mat(3, 3, CV_32F, &_camera_matrix_depth[0]);
    // vector<float> _dist_coeffs_depth = { 
    //     depth_intrinsics->param.k1, depth_intrinsics->param.k2, 
    //     depth_intrinsics->param.p1, depth_intrinsics->param.p2, 
    //     depth_intrinsics->param.k3, depth_intrinsics->param.k4,
    //     depth_intrinsics->param.k5, depth_intrinsics->param.k6
    // };
    // Mat dist_coeffs_depth = Mat(8, 1, CV_32F, &_dist_coeffs_depth[0]);
    // Mat new_camera_matrix_depth = camera_matrix_depth.clone();
    // new_camera_matrix_depth.at<float>(1,1) = new_camera_matrix_depth.at<float>(0,0);
    
    pinhole_t depth_pinhole = create_pinhole_from_xy_range(&calibration, K4A_CALIBRATION_TYPE_DEPTH);
    interpolation_t interpolation_type = INTERPOLATION_BILINEAR_DEPTH;

    k4a_image_t depth_lut;
    k4a_image_create(
        K4A_IMAGE_FORMAT_CUSTOM,
        depth_pinhole.width,
        depth_pinhole.height,
        depth_pinhole.width * (int)sizeof(coordinate_t),
        &depth_lut
    );
    create_undistortion_lut(&calibration, K4A_CALIBRATION_TYPE_DEPTH, &depth_pinhole, depth_lut, interpolation_type);

    output_pinhole("depth_undistorted_intrinsics", depth_pinhole, mkv_meta);
    output_cam_mat("rgb_undistorted_intrinsics", new_camera_matrix_rgb, mkv_meta);

    write_json(output_dirpath + "/mkv_meta.json", mkv_meta);
    if (!process_frames) {
        write_json(output_dirpath + "/flags.json", flags);
        return true;
    }

    unsigned int nthreads = std::thread::hardware_concurrency() / 2;
    // unsigned int nthreads = 2;
    progschj::ThreadPool my_pool(nthreads);
    my_pool.set_queue_size_limit(nthreads);

    std::cout << "Processing " << input_path << std::endl;

    std::string color_ext;
    if (save_jpg) {
        color_ext = ".jpg";
    } else {
        color_ext = ".png";
    }

    while (true) {
        frame_num++;
        k4a_capture_t capture_handle = nullptr;
        k4a_stream_result_t stream_result = k4a_playback_get_next_capture(playback_handle, &capture_handle);

        if (stream_result == K4A_STREAM_RESULT_EOF) {
            break;
        }

//        if (frame_num > 200) {
//            break;
//        }
	printf(">>%d\n", frame_num);
	if (frame_num == 2){
	    return 0;
	}
        if (frame_num % 1000 == 0) {
            std::cout << "frame " << frame_num << std::endl;
        }
        if (stream_result != K4A_STREAM_RESULT_SUCCEEDED) {
            std::cerr << "Stream error for clip at frame " << frame_num << std::endl;
            k4a_capture_release(capture_handle);
            continue;
        }

        std::stringstream ss;
        ss << std::setw(6) << std::setfill('0') << frame_num;
        std::string frame_num_string = ss.str();

        json frame_ts;
        if (save_ts_json) {
            frame_ts["frame_num"] = frame_num;
        }

        k4a_image_t color_image = k4a_capture_get_color_image(capture_handle);
        if (color_image != NULL) {
            if (save_ts_json) {
                frame_ts["color"] = k4a_image_get_device_timestamp_usec(color_image);
            }
            uint8_t* buffer = k4a_image_get_buffer(color_image);
            bool success = false;
            if (!buffer) {
                std::cerr << "cannot get buffer from color_image" << std::endl;
            } else if (check_resolution(
                h_color, k4a_image_get_height_pixels(color_image),
                w_color, k4a_image_get_width_pixels(color_image),
                "color"
            )) {
                success = true;
                if (save_color || save_color_undistorted) {
                    Mat color_mat(h_color, w_color, CV_8UC4, (void*)buffer, Mat::AUTO_STEP);
                    if (save_color) {
                        my_pool.enqueue(save_mat, output_color_dirpath + "/" + frame_num_string + color_ext, color_mat.clone());
                    }
                    if (save_color_undistorted) {
                        Mat color_mat_undistorted;
                        undistort(color_mat, color_mat_undistorted, camera_matrix_rgb, dist_coeffs_rgb, new_camera_matrix_rgb);
                        my_pool.enqueue(save_mat, output_color_undistorted_dirpath + "/" + frame_num_string + color_ext, color_mat_undistorted.clone());
                        color_mat_undistorted.release();
                    }
                    color_mat.release();
                }
            }
            if (!success) {
                k4a_image_release(color_image);
                color_image = NULL;
            }
        }

        k4a_image_t depth_image = k4a_capture_get_depth_image(capture_handle);
        if (depth_image != NULL) {
            if (save_ts_json) {
                frame_ts["depth"] = k4a_image_get_device_timestamp_usec(depth_image);
            }
            uint16_t* buffer = reinterpret_cast<uint16_t*>(k4a_image_get_buffer(depth_image));
            bool success = false;
            if (!buffer) {
                std::cerr << "cannot get buffer from depth_image" << std::endl;
            } else if (check_resolution(
                h_depth, k4a_image_get_height_pixels(depth_image),
                w_depth, k4a_image_get_width_pixels(depth_image),
                "depth"
            )) {
                success = true;
                if (save_depth || save_depth_undistorted) {
                    UMat depth_umat;
                    create_mat_from_buffer<uint16_t>(buffer, w_depth, h_depth).copyTo(depth_umat);
                    if (save_depth) {
                        my_pool.enqueue(save_umat, output_depth_dirpath + "/" + frame_num_string + ".png", depth_umat.clone());
                    }
                    if (save_depth_undistorted) {
                        k4a_image_t depth_image_undistorted = NULL;
                        k4a_image_create(
                            K4A_IMAGE_FORMAT_DEPTH16,
                            depth_pinhole.width,
                            depth_pinhole.height,
                            depth_pinhole.width * (int)sizeof(uint16_t),
                            &depth_image_undistorted
                        );
                        remap<uint16_t>(depth_image, depth_lut, depth_image_undistorted, interpolation_type);
                        uint16_t *buffer_undistorted = reinterpret_cast<uint16_t *>(k4a_image_get_buffer(depth_image_undistorted));
                        UMat depth_umat_undistorted;
                        create_mat_from_buffer<uint16_t>(buffer_undistorted, w_depth, h_depth).copyTo(depth_umat_undistorted);
                        my_pool.enqueue(save_umat, output_depth_undistorted_dirpath + "/" + frame_num_string + ".png", depth_umat_undistorted.clone());
                        depth_umat_undistorted.release();
                        k4a_image_release(depth_image_undistorted);
                    }
                    depth_umat.release();
                }
            }
            if (!success) {
                k4a_image_release(depth_image);
                depth_image = NULL;
            }
        }

        if (save_bt && depth_image != NULL) {
            UMat index_map_umat;
            UMat index_map_umat_undistorted;
            if (execute_bt(
                frames_tracking,
                frame_num,
                tracker, 
                capture_handle, 
                depth_lut,
                save_bt_json,
                save_bt_index_map, save_bt_index_map_undistorted,
                index_map_umat, index_map_umat_undistorted
            )) {
                if (save_bt_index_map) {
                    my_pool.enqueue(save_umat, output_bt_index_map_dirpath + "/" + frame_num_string + ".png", index_map_umat.clone());
                }
                if (save_bt_index_map_undistorted) {
                    my_pool.enqueue(save_umat, output_bt_index_map_undistorted_dirpath + "/" + frame_num_string + ".png", index_map_umat_undistorted.clone());
                }
            } else {
                std::cerr << "Predict joints failed for clip at frame " << frame_num << std::endl;
            }
            index_map_umat.release();
            index_map_umat_undistorted.release();
        }

        if (color_image != NULL && depth_image != NULL) {
            if (save_transformed_images) {
                k4a_image_t transformed_depth_image = NULL;
                k4a_image_t transformed_color_image = NULL;
                k4a_image_t depth_point_cloud = NULL;
                k4a_image_t color_point_cloud = NULL;
                if (create_transformed_images(
                    calibration,
                    transformation,
                    color_image, depth_image,
                    &transformed_depth_image, &transformed_color_image,
                    &depth_point_cloud, &color_point_cloud
                )) {
                    UMat umat;
                    uint8_t* buffer;
                    uint16_t *x, *y, *z;

                    if (save_transformed_depth || save_transformed_depth_undistorted) {
                        buffer = k4a_image_get_buffer(transformed_depth_image);
                        uint16_t *depth_buffer = reinterpret_cast<uint16_t *>(buffer);
                        create_mat_from_buffer<uint16_t>(depth_buffer, w_color, h_color).copyTo(umat);
                        if (save_transformed_depth) {
                            my_pool.enqueue(save_umat, output_transformed_depth_dirpath + "/" + frame_num_string + ".png", umat.clone());
                        }
                        // undistorted imgs look good, not checked
                        if (save_transformed_depth_undistorted) {
                            UMat umat2;
                            undistort(umat, umat2, camera_matrix_rgb, dist_coeffs_rgb, new_camera_matrix_rgb); // transformed depth image is in rgb space
                            my_pool.enqueue(save_umat, output_transformed_depth_undistorted_dirpath + "/" + frame_num_string + ".png", umat2.clone());
                            umat2.release();
                        }
                        umat.release();
                    }

                    if (save_transformed_color || save_transformed_color_undistorted) {
                        buffer = k4a_image_get_buffer(transformed_color_image);
                        Mat color_mat(h_depth, w_depth, CV_8UC4, (void*)buffer, Mat::AUTO_STEP);
                        if (save_transformed_color) {
                            my_pool.enqueue(save_mat, output_transformed_color_dirpath + "/" + frame_num_string + color_ext, color_mat.clone());
                        }
                        if (save_transformed_color_undistorted) {
                            // // undistorts, but result is too close (seems like focal lengh is too big)
                            // Mat transformed_color_undistorted;
                            // undistort(color_mat, transformed_color_undistorted, camera_matrix_depth, dist_coeffs_depth, new_camera_matrix_depth);  // transformed rgb image is in depth space
                            // my_pool.enqueue(save_mat, output_transformed_color_undistorted_dirpath + "/" + frame_num_string + ".png", transformed_color_undistorted.clone());
                            // transformed_color_undistorted.release();

                            k4a_image_t transformed_color_image_undistorted = NULL;
                            k4a_image_create(
                                K4A_IMAGE_FORMAT_COLOR_BGRA32,
                                w_depth,
                                h_depth,
                                w_depth * 4 * (int)sizeof(uint8_t),
                                &transformed_color_image_undistorted
                            );
                            remap2<uint8_t>(depth_image, transformed_color_image, depth_lut, transformed_color_image_undistorted, 4, interpolation_type);
                            buffer = k4a_image_get_buffer(transformed_color_image_undistorted);
                            Mat color_mat_undistorted(h_depth, w_depth, CV_8UC4, (void*)buffer, Mat::AUTO_STEP);
                            my_pool.enqueue(save_mat, output_transformed_color_undistorted_dirpath + "/" + frame_num_string + color_ext, color_mat_undistorted.clone());
                            color_mat_undistorted.release();
                            k4a_image_release(transformed_color_image_undistorted);
                        }
                        color_mat.release();
                    }

                    if (save_depth_point_cloud) {
                        x = new uint16_t[w_depth * h_depth];
                        y = new uint16_t[w_depth * h_depth];
                        z = new uint16_t[w_depth * h_depth];
                        split_point_cloud(depth_point_cloud, x, y, z);
                        create_mat_from_buffer<uint16_t>(x, w_depth, h_depth).copyTo(umat);
                        my_pool.enqueue(save_umat, output_depth_point_cloud_dirpath_x + "/" + frame_num_string + ".png", umat.clone());
                        umat.release();
                        create_mat_from_buffer<uint16_t>(y, w_depth, h_depth).copyTo(umat);
                        my_pool.enqueue(save_umat, output_depth_point_cloud_dirpath_y + "/" + frame_num_string + ".png", umat.clone());
                        umat.release();
                        create_mat_from_buffer<uint16_t>(z, w_depth, h_depth).copyTo(umat);
                        my_pool.enqueue(save_umat, output_depth_point_cloud_dirpath_z + "/" + frame_num_string + ".png", umat.clone());
                        umat.release();
                        delete[] x;
                        delete[] y;
                        delete[] z;
                    }

                    if (save_color_point_cloud) {
                        x = new uint16_t[w_color * h_color];
                        y = new uint16_t[w_color * h_color];
                        z = new uint16_t[w_color * h_color];
                        split_point_cloud(color_point_cloud, x, y, z);
                        create_mat_from_buffer<uint16_t>(x, w_color, h_color).copyTo(umat);
                        my_pool.enqueue(save_umat, output_color_point_cloud_dirpath_x + "/" + frame_num_string + ".png", umat.clone());
                        umat.release();
                        create_mat_from_buffer<uint16_t>(y, w_color, h_color).copyTo(umat);
                        my_pool.enqueue(save_umat, output_color_point_cloud_dirpath_y + "/" + frame_num_string + ".png", umat.clone());
                        umat.release();
                        create_mat_from_buffer<uint16_t>(z, w_color, h_color).copyTo(umat);
                        my_pool.enqueue(save_umat, output_color_point_cloud_dirpath_z + "/" + frame_num_string + ".png", umat.clone());
                        umat.release();
                        delete[] x;
                        delete[] y;
                        delete[] z;
                    }
                } else {
                    std::cerr << "create_transformed_images error" << std::endl;
                }
                if (transformed_depth_image != NULL) {
                    k4a_image_release(transformed_depth_image);
                }
                if (transformed_color_image != NULL) {
                    k4a_image_release(transformed_color_image);
                }
                if (depth_point_cloud != NULL) {
                    k4a_image_release(depth_point_cloud);
                }
                if (color_point_cloud != NULL) {
                    k4a_image_release(color_point_cloud);
                }
            }
        }
        
        if (color_image != NULL) {
            k4a_image_release(color_image);
        }
        if (depth_image != NULL) {
            k4a_image_release(depth_image);
        }
        if (save_ts_json) {
            frames_ts.push_back(frame_ts);
        }
        k4a_capture_release(capture_handle);
    }

    std::cout << std::endl << "DONE" << std::endl;

    std::cout << "Total read " << frame_num << " frames" << std::endl;
    
    if (save_ts_json) {
        write_json(output_dirpath + "/ts.json", frames_ts);
    }
    if (save_bt_json) {
        write_json(output_dirpath + "/bt.json", frames_tracking);
    }
    write_json(output_dirpath + "/flags.json", flags);

    k4a_playback_close(playback_handle);

    return true;
}

int main(int argc, char **argv) {

    // std::cout << "OpenCV version : " << CV_VERSION << std::endl;
    // std::cout << "Major version : " << CV_MAJOR_VERSION << std::endl;
    // std::cout << "Minor version : " << CV_MINOR_VERSION << std::endl;
    // std::cout << "Subminor version : " << CV_SUBMINOR_VERSION << std::endl;

    const size_t num_args = 19;
    if (argc != num_args) {
        std::cerr << "usage: ./offline_processor2 <input_mkv_filepath> <output_dirpath> <bool_flag_1> <bool_flag_2> .. <bool_flag_14> <rotation>" << std::endl;
        std::cerr << "got " << argc << " args, need " << num_args << " args" << std::endl;
        return -1;
    }

    bool save_color = atoi(argv[3]);
    bool save_color_undistorted = atoi(argv[4]);
    bool save_depth = atoi(argv[5]);
    bool save_depth_undistorted = atoi(argv[6]);
    bool save_ts_json = atoi(argv[7]);
    bool save_bt_json = atoi(argv[8]);
    bool save_bt_index_map = atoi(argv[9]);
    bool save_bt_index_map_undistorted = atoi(argv[10]);
    bool save_transformed_depth = atoi(argv[11]);
    bool save_transformed_depth_undistorted = atoi(argv[12]);
    bool save_transformed_color = atoi(argv[13]);
    bool save_transformed_color_undistorted = atoi(argv[14]);
    bool save_depth_point_cloud = atoi(argv[15]);
    bool save_color_point_cloud = atoi(argv[16]);

    int rotate = atoi(argv[17]);
    bool save_jpg = atoi(argv[18]);

    return process_mkv_offline(
        argv[1], argv[2],
        save_color,
        save_color_undistorted,
        save_depth,
        save_depth_undistorted,
        save_ts_json,
        save_bt_json,
        save_bt_index_map,
        save_bt_index_map_undistorted,
        save_transformed_depth,
        save_transformed_depth_undistorted,
        save_transformed_color,
        save_transformed_color_undistorted,
        save_depth_point_cloud,
        save_color_point_cloud,
        rotate,
        save_jpg
    ) ? 0 : -1;
}
