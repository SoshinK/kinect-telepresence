import numpy as np
import cv2 as cv2

def create_undistortion_lut(depth_shape, intrinsic_matrix, 
                            rot_vec, t_vec,
                            dist_coefs, interpolation_type):
    ray = np.array([0., 0., 1.])
    assert len(depth_shape) == 2
    
    width, height = depth_shape
    fx, fy = intrinsic_matrix[0, 0], intrinsic_matrix[1, 1]
    px, py = intrinsic_matrix[0, 2]. intrinsic_matrix[1, 2]

    lut_data = []

    # idx = 0

    for y in range(height):
        ray[1] = (float(y) - py) / fy
        for x in range(width):
            ray[0] = (float(x) - px) / fx

            distorted, _ = cv2.projectPoints([ray], rot_vec, t_vec, intrinsic_matrix, dist_coefs)
            print("dist shape: ", distorted.shape)

            src = np.array([0, 0])

            if interpolation_type == 'nearest':
                src[0] = int(np.floor(distorted[0] + 0.5))
                src[1] = int(np.floor(distorted[1] + 0.5))
            elif interpolation_type == 'bilinear':
                src[0] = int(np.floor(distorted[0]))
                src[1] = int(np.floor(distorted[1]))
            else:
                ValueError("Unknown interpolation type")
            
            if src[0] >= 0 and src[0] < width and src[1] >= 0 and src[1] < height:
                lut_data[idx] = src
            
