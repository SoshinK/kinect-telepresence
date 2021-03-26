//
// Created by vakhitov on 03.12.19.
//

#include "undistort.h"

#include <tgmath.h>

void compute_xy_range(const k4a_calibration_t* calibration,
                             const k4a_calibration_type_t camera,
                             const int width,
                             const int height,
                             float& x_min,
                             float& x_max,
                             float& y_min,
                             float& y_max)
{
    // Step outward from the centre point until we find the bounds of valid projection
    const float step_u = 0.25f;
    const float step_v = 0.25f;
    const float min_u = 0;
    const float min_v = 0;
    const float max_u = (float)width - 1;
    const float max_v = (float)height - 1;
    const float center_u = 0.5f * width;
    const float center_v = 0.5f * height;

    int valid;
    k4a_float2_t p;
    k4a_float3_t ray;

    // search x_min
    for (float uv[2] = { center_u, center_v }; uv[0] >= min_u; uv[0] -= step_u)
    {
        p.xy.x = uv[0];
        p.xy.y = uv[1];
        k4a_calibration_2d_to_3d(calibration, &p, 1.f, camera, camera, &ray, &valid);

        if (!valid)
        {
            break;
        }
        x_min = ray.xyz.x;
    }

    // search x_max
    for (float uv[2] = { center_u, center_v }; uv[0] <= max_u; uv[0] += step_u)
    {
        p.xy.x = uv[0];
        p.xy.y = uv[1];
        k4a_calibration_2d_to_3d(calibration, &p, 1.f, camera, camera, &ray, &valid);

        if (!valid)
        {
            break;
        }
        x_max = ray.xyz.x;
    }

    // search y_min
    for (float uv[2] = { center_u, center_v }; uv[1] >= min_v; uv[1] -= step_v)
    {
        p.xy.x = uv[0];
        p.xy.y = uv[1];
        k4a_calibration_2d_to_3d(calibration, &p, 1.f, camera, camera, &ray, &valid);

        if (!valid)
        {
            break;
        }
        y_min = ray.xyz.y;
    }

    // search y_max
    for (float uv[2] = { center_u, center_v }; uv[1] <= max_v; uv[1] += step_v)
    {
        p.xy.x = uv[0];
        p.xy.y = uv[1];
        k4a_calibration_2d_to_3d(calibration, &p, 1.f, camera, camera, &ray, &valid);

        if (!valid)
        {
            break;
        }
        y_max = ray.xyz.y;
    }
}

pinhole_t create_pinhole_from_xy_range(const k4a_calibration_t* calibration, const k4a_calibration_type_t camera)
{
    int width = calibration->depth_camera_calibration.resolution_width;
    int height = calibration->depth_camera_calibration.resolution_height;
    if (camera == K4A_CALIBRATION_TYPE_COLOR)
    {
        width = calibration->color_camera_calibration.resolution_width;
        height = calibration->color_camera_calibration.resolution_height;
    }

    float x_min = 0, x_max = 0, y_min = 0, y_max = 0;
    compute_xy_range(calibration, camera, width, height, x_min, x_max, y_min, y_max);

    pinhole_t pinhole;

    float fx = 1.f / (x_max - x_min);
    float fy = 1.f / (y_max - y_min);
    float px = -x_min * fx;
    float py = -y_min * fy;

    pinhole.fx = fx * width;
    pinhole.fy = fy * height;
    pinhole.px = px * width;
    pinhole.py = py * height;
    pinhole.width = width;
    pinhole.height = height;

    return pinhole;
}


void create_undistortion_lut(const k4a_calibration_t* calibration,
                                    const k4a_calibration_type_t camera,
                                    const pinhole_t* pinhole,
                                    k4a_image_t lut,
                                    interpolation_t type)
{
    coordinate_t* lut_data = (coordinate_t*)(void*)k4a_image_get_buffer(lut);

    k4a_float3_t ray;
    ray.xyz.z = 1.f;

    int src_width = calibration->depth_camera_calibration.resolution_width;
    int src_height = calibration->depth_camera_calibration.resolution_height;
    if (camera == K4A_CALIBRATION_TYPE_COLOR)
    {
        src_width = calibration->color_camera_calibration.resolution_width;
        src_height = calibration->color_camera_calibration.resolution_height;
    }

    for (int y = 0, idx = 0; y < pinhole->height; y++)
    {
        ray.xyz.y = ((float)y - pinhole->py) / pinhole->fy;

        for (int x = 0; x < pinhole->width; x++, idx++)
        {
            ray.xyz.x = ((float)x - pinhole->px) / pinhole->fx;

            k4a_float2_t distorted;
            int valid;
            k4a_calibration_3d_to_2d(calibration, &ray, camera, camera, &distorted, &valid);
	    /*//=========
	    printf("\n==========\n");
	    printf("px: %f py: %f\n", pinhole->px, pinhole->py);
	    printf("fx: %f fy: %f\n", pinhole->fx, pinhole->fy);
	    printf("src wh: %u %u; wh: %u %u \n", src_width, src_height, pinhole->width, pinhole->height);
	    printf("%u\n", calibration->depth_camera_calibration.intrinsics.parameter_count);
	    printf("%f %f %f %f %f  %f %f %f %f %f %f %f %f %f -empty-\n",  calibration->depth_camera_calibration.intrinsics.parameters.v[0],
			    calibration->depth_camera_calibration.intrinsics.parameters.v[1],
			    calibration->depth_camera_calibration.intrinsics.parameters.v[2],
			    calibration->depth_camera_calibration.intrinsics.parameters.v[3],
			    calibration->depth_camera_calibration.intrinsics.parameters.v[4],
			    calibration->depth_camera_calibration.intrinsics.parameters.v[5],
			    calibration->depth_camera_calibration.intrinsics.parameters.v[6],
			    calibration->depth_camera_calibration.intrinsics.parameters.v[7],
			    calibration->depth_camera_calibration.intrinsics.parameters.v[8],
			    calibration->depth_camera_calibration.intrinsics.parameters.v[9],
			    calibration->depth_camera_calibration.intrinsics.parameters.v[10],
			    calibration->depth_camera_calibration.intrinsics.parameters.v[11],
  			    calibration->depth_camera_calibration.intrinsics.parameters.v[12],
			    calibration->depth_camera_calibration.intrinsics.parameters.v[13]
			    
			    
			    );
			
	    //=========*/
            coordinate_t src;
            if (type == INTERPOLATION_NEARESTNEIGHBOR)
            {
                // Remapping via nearest neighbor interpolation
                src.x = (int)floor(distorted.xy.x + 0.5f);
                src.y = (int)floor(distorted.xy.y + 0.5f);
            }
            else if (type == INTERPOLATION_BILINEAR || type == INTERPOLATION_BILINEAR_DEPTH)
            {
                // Remapping via bilinear interpolation
                src.x = (int)floor(distorted.xy.x);
                src.y = (int)floor(distorted.xy.y);
            }
            else
            {
                printf("Unexpected interpolation type!\n");
                exit(-1);
            }
	    

            if (valid && src.x >= 0 && src.x < src_width && src.y >= 0 && src.y < src_height)
            {
                lut_data[idx] = src;

                if (type == INTERPOLATION_BILINEAR || type == INTERPOLATION_BILINEAR_DEPTH)
                {
                    // Compute the floating point weights, using the distance from projected point src to the
                    // image coordinate of the upper left neighbor
                    float w_x = distorted.xy.x - src.x;
                    float w_y = distorted.xy.y - src.y;
                    float w0 = (1.f - w_x) * (1.f - w_y);
                    float w1 = w_x * (1.f - w_y);
                    float w2 = (1.f - w_x) * w_y;
                    float w3 = w_x * w_y;

                    // Fill into lut
                    lut_data[idx].weight[0] = w0;
                    lut_data[idx].weight[1] = w1;
                    lut_data[idx].weight[2] = w2;
                    lut_data[idx].weight[3] = w3;
                }
            }
            else
            {
                lut_data[idx].x = INVALID;
                lut_data[idx].y = INVALID;
            }
	    //printf("%d %d ray: %f %f %f; distorted: %f %f; src: %d %d lut:%f %f %f %f", x, y, ray.xyz.x, ray.xyz.y, ray.xyz.z, distorted.xy.x, distorted.xy.y, src.x, src.y, lut_data[idx].weight[0], lut_data[idx].weight[1], lut_data[idx].weight[2], lut_data[idx].weight[3]);
        }
    }
}

template <typename T>
void remap(const k4a_image_t src, const k4a_image_t lut, k4a_image_t dst, interpolation_t type)
{
    int src_width = k4a_image_get_width_pixels(src);
    int dst_width = k4a_image_get_width_pixels(dst);
    int dst_height = k4a_image_get_height_pixels(dst);

    T* src_data = (T*)(void*)k4a_image_get_buffer(src);
    T* dst_data = (T*)(void*)k4a_image_get_buffer(dst);
    coordinate_t* lut_data = (coordinate_t*)(void*)k4a_image_get_buffer(lut);

    memset(dst_data, 0, (size_t)dst_width * (size_t)dst_height * sizeof(T));

    for (int i = 0; i < dst_width * dst_height; i++)
    {
	//printf(">> i: %d; %d;%d %d  weights: %f %f %f %f src_width: %d\n", i, src_data[i], lut_data[i].x, lut_data[i].y, lut_data[i].weight[0], lut_data[i].weight[1], lut_data[i].weight[2], lut_data[i].weight[3], src_width);

        if (lut_data[i].x != INVALID && lut_data[i].y != INVALID)
        {
            if (type == INTERPOLATION_NEARESTNEIGHBOR)
            {
                dst_data[i] = src_data[lut_data[i].y * src_width + lut_data[i].x];
            }
            else if (type == INTERPOLATION_BILINEAR || type == INTERPOLATION_BILINEAR_DEPTH)
            {
                const T neighbors[4]{ src_data[lut_data[i].y * src_width + lut_data[i].x],
                                             src_data[lut_data[i].y * src_width + lut_data[i].x + 1],
                                             src_data[(lut_data[i].y + 1) * src_width + lut_data[i].x],
                                             src_data[(lut_data[i].y + 1) * src_width + lut_data[i].x + 1] };

                // If the image contains invalid data, e.g. depth image contains value 0, ignore the bilinear
                // interpolation for current target pixel if one of the neighbors contains invalid data to avoid
                // introduce noise on the edge. If the image is color or ir images, user should use
                // INTERPOLATION_BILINEAR
                if (type == INTERPOLATION_BILINEAR_DEPTH)
                {
                    // If the image contains invalid data, e.g. depth image contains value 0, ignore the bilinear
                    // interpolation for current target pixel if one of the neighbors contains invalid data to avoid
                    // introduce noise on the edge. If the image is color or ir images, user should use
                    // INTERPOLATION_BILINEAR
                    if (neighbors[0] == 0 || neighbors[1] == 0 || neighbors[2] == 0 || neighbors[3] == 0)
                    {
			//printf("here1(\n");
                        continue;
                    }

                    // Ignore interpolation at large depth discontinuity without disrupting slanted surface
                    // Skip interpolation threshold is estimated based on the following logic:
                    // - angle between two pixels is: theta = 0.234375 degree (120 degree / 512) in binning resolution
                    // mode
                    // - distance between two pixels at same depth approximately is: A ~= sin(theta) * depth
                    // - distance between two pixels at highly slanted surface (e.g. alpha = 85 degree) is: B = A /
                    // cos(alpha)
                    // - skip_interpolation_ratio ~= sin(theta) / cos(alpha)
                    // We use B as the threshold that to skip interpolation if the depth difference in the triangle is
                    // larger than B. This is a conservative threshold to estimate largest distance on a highly slanted
                    // surface at given depth, in reality, given distortion, distance, resolution difference, B can be
                    // smaller
                    const float skip_interpolation_ratio = 0.04693441759f;
                    float depth_min = min(min(neighbors[0], neighbors[1]), min(neighbors[2], neighbors[3]));
                    float depth_max = max(max(neighbors[0], neighbors[1]), max(neighbors[2], neighbors[3]));
                    float depth_delta = depth_max - depth_min;
                    float skip_interpolation_threshold = skip_interpolation_ratio * depth_min;
                    if (depth_delta > skip_interpolation_threshold)
                    {
			//printf("here2(\n");
                        continue;
                    }
                }

                dst_data[i] = (T)(neighbors[0] * lut_data[i].weight[0] + neighbors[1] * lut_data[i].weight[1] +
                                         neighbors[2] * lut_data[i].weight[2] + neighbors[3] * lut_data[i].weight[3] +
                                         0.5f);
		//printf("!! i: %d; dst: %d; neigh: %d %d %d %d weights: %f %f %f %f\n", i, dst_data[i], neighbors[0], neighbors[1], neighbors[2], neighbors[3], lut_data[i].weight[0], lut_data[i].weight[1], lut_data[i].weight[2], lut_data[i].weight[3]);
            }
            else
            {
                printf("Unexpected interpolation type!\n");
                exit(-1);
            }
        }
    }
}

template void remap<uint16_t>(const k4a_image_t src, const k4a_image_t lut, k4a_image_t dst, interpolation_t type);
template void remap<uint8_t>(const k4a_image_t src, const k4a_image_t lut, k4a_image_t dst, interpolation_t type);


template <typename T>
void remap2(const k4a_image_t depth_src, const k4a_image_t src, const k4a_image_t lut, k4a_image_t dst, const size_t channels, interpolation_t type)
{
    int src_width = k4a_image_get_width_pixels(src);
    int dst_width = k4a_image_get_width_pixels(dst);
    int dst_height = k4a_image_get_height_pixels(dst);

    uint16_t* depth_src_data = (uint16_t*)(void*)k4a_image_get_buffer(depth_src);
    T* src_data = (T*)(void*)k4a_image_get_buffer(src);
    T* dst_data = (T*)(void*)k4a_image_get_buffer(dst);
    coordinate_t* lut_data = (coordinate_t*)(void*)k4a_image_get_buffer(lut);

    memset(dst_data, 0, (size_t)dst_width * (size_t)dst_height * sizeof(T) * channels);

    for (int i = 0; i < dst_width * dst_height; i++)
    {
        if (lut_data[i].x != INVALID && lut_data[i].y != INVALID)
        {
            if (type == INTERPOLATION_NEARESTNEIGHBOR)
            {
                dst_data[i] = src_data[lut_data[i].y * src_width + lut_data[i].x];
            }
            else if (type == INTERPOLATION_BILINEAR || type == INTERPOLATION_BILINEAR_DEPTH)
            {
                const size_t i00 = lut_data[i].y * src_width + lut_data[i].x;
                const size_t i01 = lut_data[i].y * src_width + lut_data[i].x + 1;
                const size_t i10 = (lut_data[i].y + 1) * src_width + lut_data[i].x;
                const size_t i11 = (lut_data[i].y + 1) * src_width + lut_data[i].x + 1;
                T neighbors_values[channels][4];
                for (size_t channel = 0; channel < channels; channel++) {
                    neighbors_values[channel][0] = src_data[channels * i00 + channel];
                    neighbors_values[channel][1] = src_data[channels * i01 + channel];
                    neighbors_values[channel][2] = src_data[channels * i10 + channel];
                    neighbors_values[channel][3] = src_data[channels * i11 + channel];
                }
                uint16_t neighbors[4];
                neighbors[0] = depth_src_data[i00];
                neighbors[1] = depth_src_data[i01];
                neighbors[2] = depth_src_data[i10];
                neighbors[3] = depth_src_data[i11];

                // If the image contains invalid data, e.g. depth image contains value 0, ignore the bilinear
                // interpolation for current target pixel if one of the neighbors contains invalid data to avoid
                // introduce noise on the edge. If the image is color or ir images, user should use
                // INTERPOLATION_BILINEAR
                if (type == INTERPOLATION_BILINEAR_DEPTH)
                {
                    // If the image contains invalid data, e.g. depth image contains value 0, ignore the bilinear
                    // interpolation for current target pixel if one of the neighbors contains invalid data to avoid
                    // introduce noise on the edge. If the image is color or ir images, user should use
                    // INTERPOLATION_BILINEAR
                    if (neighbors[0] == 0 || neighbors[1] == 0 || neighbors[2] == 0 || neighbors[3] == 0)
                    {
                        continue;
                    }

                    // Ignore interpolation at large depth discontinuity without disrupting slanted surface
                    // Skip interpolation threshold is estimated based on the following logic:
                    // - angle between two pixels is: theta = 0.234375 degree (120 degree / 512) in binning resolution
                    // mode
                    // - distance between two pixels at same depth approximately is: A ~= sin(theta) * depth
                    // - distance between two pixels at highly slanted surface (e.g. alpha = 85 degree) is: B = A /
                    // cos(alpha)
                    // - skip_interpolation_ratio ~= sin(theta) / cos(alpha)
                    // We use B as the threshold that to skip interpolation if the depth difference in the triangle is
                    // larger than B. This is a conservative threshold to estimate largest distance on a highly slanted
                    // surface at given depth, in reality, given distortion, distance, resolution difference, B can be
                    // smaller
                    const float skip_interpolation_ratio = 0.04693441759f;
                    float depth_min = min(min(neighbors[0], neighbors[1]), min(neighbors[2], neighbors[3]));
                    float depth_max = max(max(neighbors[0], neighbors[1]), max(neighbors[2], neighbors[3]));
                    float depth_delta = depth_max - depth_min;
                    float skip_interpolation_threshold = skip_interpolation_ratio * depth_min;
                    if (depth_delta > skip_interpolation_threshold)
                    {
                        continue;
                    }
                }
                for (size_t channel = 0; channel < channels; channel++) {
                    dst_data[channels * i + channel] = (T)(
                        neighbors_values[channel][0] * lut_data[i].weight[0] + 
                        neighbors_values[channel][1] * lut_data[i].weight[1] +
                        neighbors_values[channel][2] * lut_data[i].weight[2] + 
                        neighbors_values[channel][3] * lut_data[i].weight[3] +
                        0.5f
                    );
                }
            }
            else
            {
                printf("Unexpected interpolation type!\n");
                exit(-1);
            }
        }
    }
}

template void remap2<uint16_t>(const k4a_image_t depth_src, const k4a_image_t src, const k4a_image_t lut, k4a_image_t dst, const size_t channels, interpolation_t type);
template void remap2<uint8_t>(const k4a_image_t depth_src, const k4a_image_t src, const k4a_image_t lut, k4a_image_t dst, const size_t channels, interpolation_t type);


template<typename T> Mat create_mat_from_buffer(T *data, int width, int height, int channels)
{
    Mat mat(height, width, CV_MAKETYPE(DataType<T>::type, channels));
    memcpy(mat.data, data, width * height * channels * sizeof(T));
    return mat;
}

template Mat create_mat_from_buffer<uint16_t>(uint16_t *data, int width, int height, int channels);
template Mat create_mat_from_buffer<uint8_t>(uint8_t *data, int width, int height, int channels);
