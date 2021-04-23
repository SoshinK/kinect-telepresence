'''
class DistortionRectifier
'''
import numpy as np
import cv2 as cv2

from kinect_telepresence.camera.intrinsics import Intrinsics
from kinect_telepresence.utils.utils import check_mtrx

class DistortionRectifier:
    def __init__(self, intrinsics: Intrinsics, interpolation_type):
        self._intrinsics = intrinsics
        self._interpolation_type = interpolation_type
    def __call__(self, img):
        pass


class RgbDistortionRectifier(DistortionRectifier):
    def __call__(self, img):
        return cv2.undistort(img.copy(), self._intrinsics.cam_mtrx_dist, self._intrinsics.dist_coeffs,  newCameraMatrix=self._intrinsics.cam_mtrx_undist)


class DepthDistortionRectifier(DistortionRectifier):

    INVALID_LUT_DATA = -1111

    def __init__(self, intrinsics: Intrinsics, interpolation_type):
        super().__init__(intrinsics, interpolation_type)
        self._undistortion_lut = DepthDistortionRectifier._create_undistortion_lut(self._intrinsics, self._interpolation_type)


    def _create_undistortion_lut(intrinsics: Intrinsics, interpolation_type):
        height, width = intrinsics.resolution
        fx, fy = intrinsics.cam_mtrx_undist[0, 0], intrinsics.cam_mtrx_undist[1, 1]
        px, py = intrinsics.cam_mtrx_undist[0, 2], intrinsics.cam_mtrx_undist[1, 2]

        ii, jj = np.meshgrid(np.arange(width), np.arange(height))

        rays = np.array([(np.float32(ii.ravel()) - px) / fx,
                (np.float32(jj.ravel()) - py) / fy,
                np.ones(width * height, dtype=np.float32)])

        distorted, _ = cv2.projectPoints(rays, np.zeros(3), np.zeros(3), intrinsics.cam_mtrx_dist, intrinsics.dist_coeffs)
        distorted = distorted[:, 0, :].T # (2, N)
        # print("dist", distorted.shape, rays.shape)#, y, x)
        # exit()
        # src = np.array([0, 0])

        src = np.array([])

        if interpolation_type == 'nearest':
            src = np.array([np.int32(np.floor(distorted[0] + 0.5)), np.int32(np.floor(distorted[1] + 0.5))])
        elif interpolation_type == 'bilinear':
            src = np.array([np.int32(np.floor(distorted[0])), np.int32(np.floor(distorted[1]))])
        else:
            ValueError("Unknown interpolation type")
        
        lut_data = np.array([[0, 0] for _ in range(height * width)]).T
        # print("!!", lut_data.shape)
        mask1 = src[0] >= 0
        mask2 = src[1] >= 0
        mask3 = src[0] < width
        mask4 = src[1] < height

        mask = mask1 & mask2 & mask3 & mask4
        
        lut_data[:, mask] = src[:, mask]
        lut_data[:, ~mask] = np.array([[
            DepthDistortionRectifier.INVALID_LUT_DATA, 
            DepthDistortionRectifier.INVALID_LUT_DATA] for _ in range(mask[~mask].shape[0])]).T
        
        if interpolation_type == 'bilinear':
            weights = np.array([[0., 0., 0., 0.] for _ in range(height * width)]).T
            weights[:, ~mask] = np.array([[
                DepthDistortionRectifier.INVALID_LUT_DATA, 
                DepthDistortionRectifier.INVALID_LUT_DATA, 
                DepthDistortionRectifier.INVALID_LUT_DATA, 
                DepthDistortionRectifier.INVALID_LUT_DATA] for _ in range(mask[~mask].shape[0])]).T

            w_x = distorted[0][mask] - src[0][mask]
            w_y = distorted[1][mask] - src[1][mask]
            w0 = (1 - w_x) * (1 - w_y)
            w1 = w_x * (1 - w_y)
            w2 = (1 - w_x) * w_y
            w3 = w_x * w_y 
            
            weights[0, mask] = w0
            weights[1, mask] = w1
            weights[2, mask] = w2
            weights[3, mask] = w3 
            lut_data = np.append(lut_data, weights, axis=0)
        return lut_data.T

    def __call__(self, depthmap):
        check_mtrx(depthmap, "depthmap", self._intrinsics.resolution)

        height, width = self._intrinsics.resolution
        
        depthmap_flattened = depthmap.reshape((-1))
        
        assert depthmap_flattened.shape[0] == len(self._undistortion_lut)

        dst_img = np.zeros_like(depthmap_flattened)

        mask = self._undistortion_lut[:, 0] != DepthDistortionRectifier.INVALID_LUT_DATA
        # print(self._undistortion_lut.shape, depthmap_flattened.shape, self._undistortion_lut[mask].shape)
        if self._interpolation_type == 'nearest':
            dst_img[mask] = depthmap_flattened[self._undistortion_lut[mask:, 1] * width + self._undistortion_lut[mask:, 0]]
        elif self._interpolation_type == 'bilinear':
            # print(depthmap_flattened[np.int32(self._undistortion_lut[mask, 1] * width + self._undistortion_lut[mask, 0])])
            # exit()
            neighbors = np.zeros((depthmap_flattened.shape[0], 4))
            neighbors[mask] = np.array([depthmap_flattened[np.int32(self._undistortion_lut[mask, 1] * width + self._undistortion_lut[mask, 0])],
                                depthmap_flattened[np.int32(self._undistortion_lut[mask, 1] * width + self._undistortion_lut[mask, 0] + 1)],
                                depthmap_flattened[np.int32((self._undistortion_lut[mask, 1] + 1) * width + self._undistortion_lut[mask, 0])],
                                depthmap_flattened[np.int32((self._undistortion_lut[mask, 1] + 1) * width + self._undistortion_lut[mask, 0] + 1)]
                                ]).T
            idxs_where_eq_zero = np.unique(np.argwhere(neighbors == 0)[:, 0])
            mask_where_not_eq_zero = np.ones(depthmap_flattened.shape[0], dtype=bool)
            mask_where_not_eq_zero[idxs_where_eq_zero] = False
            # print(mask_where_not_eq_zero[mask_where_not_eq_zero == True].shape)
            
            mask_where_calc = mask_where_not_eq_zero
            dst_img[mask_where_calc] = np.sum(neighbors[mask_where_calc] * self._undistortion_lut[mask_where_calc, 2:], axis=1) + 0.5

        return dst_img.reshape(depthmap.shape)


