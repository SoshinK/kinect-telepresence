import numpy as np
import cv2 as cv2
import matplotlib.pyplot as plt

INVALID_LUT_DATA = -1111

def create_undistortion_lut(depth_shape, intrinsic_matrix, intrinsic_matrix_undist, 
                            rot_vec, t_vec,
                            dist_coefs, interpolation_type):
    ray = np.array([0., 0., 1.])
    assert len(depth_shape) == 2
    
    height, width = depth_shape
    fx, fy = intrinsic_matrix_undist[0, 0], intrinsic_matrix_undist[1, 1]
    px, py = intrinsic_matrix_undist[0, 2], intrinsic_matrix_undist[1, 2]

    lut_data = []

    # idx = 0

    for y in range(height):
        ray[1] = (float(y) - py) / fy
        for x in range(width):
            ray[0] = (float(x) - px) / fx

            distorted, _ = cv2.projectPoints(np.array([ray]), rot_vec, t_vec, intrinsic_matrix, dist_coefs)
            distorted = distorted[0, 0]
            # print("dist", distorted, ray, y, x)

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
                weights = [src[0], src[1]]

                if interpolation_type == 'bilinear':
                    w_x = distorted[0] - src[0]
                    w_y = distorted[1] - src[1]
                    w0 = (1 - w_x) * (1 - w_y)
                    w1 = w_x * (1 - w_y)
                    w2 = (1 - w_x) * w_y
                    w3 = w_x * w_y 

                    weights.append(w0)
                    weights.append(w1)
                    weights.append(w2)
                    weights.append(w3)

                lut_data.append(weights)
            else:
                lut_data.append(INVALID_LUT_DATA)
            # print(x, y, "ray: ", ray, "distorted: ", distorted, "src: ", src, "lut_data: ", lut_data[-1])
    return lut_data


def remap(src_img, lut, interpolation_type):
    assert len(src_img.shape) == 2
    height, width = src_img.shape
    
    src_img_flattened = src_img.reshape((-1))
    
    assert src_img_flattened.shape[0] == len(lut)

    dst_img = np.zeros_like(src_img_flattened)

    for i in range(src_img_flattened.shape[0]):
        # print(">>", i, src_img_flattened[i], lut[i], width)
        if lut[i] != INVALID_LUT_DATA:

            if interpolation_type == 'nearest':
                dst_img[i] = src_img_flattened[lut[i][1] * width + lut[i][0]]
            elif interpolation_type == 'bilinear':

                neighbors = [src_img_flattened[lut[i][1] * width + lut[i][0]],
                            src_img_flattened[lut[i][1] * width + lut[i][0] + 1],
                            src_img_flattened[(lut[i][1] + 1) * width + lut[i][0]],
                            src_img_flattened[(lut[i][1] + 1) * width + lut[i][0] + 1]
                            ]
                if neighbors[0] == 0 or neighbors[1] == 0 or neighbors[2] == 0 or neighbors[3] == 0:
                    # print("here1(")
                    continue
                skip_interpolation_ratio = 0.04693441759
                depth_min = np.amin(neighbors)
                depth_max = np.amax(neighbors)
                depth_delta = depth_max - depth_min
                skip_interpolation_threshold = skip_interpolation_ratio * depth_min
                if depth_delta > skip_interpolation_threshold:
                    # print("here2(")
                    continue

                dst_img[i] = neighbors[0] * lut[i][2] + \
                            neighbors[1] * lut[i][3] + \
                            neighbors[2] * lut[i][4] + \
                            neighbors[3] * lut[i][5] + 0.5
                # print("!!", i, dst_img[i], neighbors, lut[i])
            else:
                ValueError("Unknown interpolation type")
    return dst_img.reshape(src_img.shape)

def undistort(distorted, intrinsic_mat, intrinsic_mat_undist, dist_coefs, interpolation_type='bilinear'):
    lut = create_undistortion_lut(distorted.shape, 
                                intrinsic_mat, 
                                intrinsic_mat_undist, 
                                np.array([0., 0., 0.]), 
                                np.array([0., 0., 0.]), 
                                dist_coefs, 
                                interpolation_type)
    undist = remap(distorted.copy(), lut, interpolation_type)
    return undist

def main():
    distorted = cv2.imread('depth0177.png', -1)
    gt = cv2.imread('000177.png', -1)
    plt.imshow(distorted)
    plt.show()

    intr = np.array([[505.05084228515625,                 0, 338.0733642578125],
                    [                  0, 504.9956359863281, 338.06817626953125],
                    [0,                                   0,                  1]])
    intr_undist = np.array([[430.94134521484375,                 0, 345.10784912109375],
                    [                  0, 441.03570556640625, 351.9667053222656],
                    [0,                                   0,                  1]])
    dist_coefs = np.array([5.325788497924805, 3.2844455242156982, -2.5223982902389253e-06, -8.48980707814917e-05, 0.1552966833114624, 5.6506853103637695, 5.049447536468506, 0.858729898929596])

    # r_vec = np.array([
    #         -0.10038138180971146,
    #         -0.0006439610733650625,
    #         0.002789221005514264
    #     ])
    r_vec = np.array([
            0.,
            0.,
            0.
        ])
    t_vec = np.array([0.,
            0.,
            0.])

    lut = create_undistortion_lut(distorted.shape, intr, intr_undist, r_vec, t_vec, dist_coefs, 'bilinear')
    # print(np.amax(lut), np.amin(lut))
    
    undist = remap(distorted.copy(), lut, 'bilinear')
    plt.subplot(131)
    plt.imshow(undist)

    plt.subplot(132)
    plt.imshow(gt)

    plt.subplot(133)
    plt.imshow(gt - undist)

    plt.show()


if __name__ == '__main__':
    main()
