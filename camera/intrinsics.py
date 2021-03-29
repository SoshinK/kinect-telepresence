'''
class Intrinsics
Stores intrinsic camera parameters, such as camera matrix and distortion coefficients
'''

import numpy as np

class Intrinsics:
    def __init__(self, cam_mtrx_dist, cam_mtrx_undist=None, dist_coeffs=None):
        self._cam_mtrx_dist = np.array(cam_mtrx_dist)
        _check_mtrx(self._cam_mtrx_dist, 'intrinsic matrix')

        self._cam_mtrx_undist = cam_mtrx_undist
        if cam_mtrx_dist not is None:
            self._cam_mtrx_undist = np.array(self._cam_mtrx_undist)
            _check_mtrx(self._cam_mtrx_undist, 'intrinsic undistorted matrix', recquired_shape = (3, 3))

            

    def _check_mtrx(mtrx, what_mtrx = "", recquired_shape = (3, 3)):
        if type(mtrx) != np.ndarray:
            raise TypeError(f"Wrong matrix type. {what_mtrx} is not np.ndarray")
        if mtrx.shape != recquired_shape:
            raise ValueError(f"Wrong matrix. {what_mtrx} should have {recquired_shape} shape")
        