'''
Roto-translation in 4d homogenious coordinates
'''

import numpy as np
import cv2 as cv2

from kinect_telepresence.geometry.pointcloud import PointCloud
from kinect_telepresence.utils.utils import check_mtrx

class RotoTranslate:

    rt_transform_default_dtype = np.float32

    def __init__(self, rot_mtrx_4x4, t_vec_4d):
        self._rot_mat = np.array(rot_mtrx_4x4).astype(RotoTranslate.rt_transform_default_dtype)
        self._t_vec = np.array(t_vec_4d).astype(RotoTranslate.rt_transform_default_dtype)
        check_mtrx(self._rot_mat, what_mtrx='4x4 rotation matrix', required_shape=(4, 4))
        check_mtrx(self._t_vec, what_mtrx='4d translation vector', required_shape=(4,))

        self._inv_rot_mat = np.linalg.inv(self._rot_mat)
        self._inv_t_vec = - self._inv_rot_mat @ self._t_vec

    @property
    def inv(self):
        return RotoTranslate(self._inv_rot_mat, self._inv_t_vec)

    @classmethod
    def from_mat_vec(cl, rot_mtrx, t_vec):
        rot_mtrx = np.array(rot_mtrx).astype(RotoTranslate.rt_transform_default_dtype) 
        t_vec = np.array(t_vec).astype(RotoTranslate.rt_transform_default_dtype)
        check_mtrx(rot_mtrx, what_mtrx='3x3 rotation matrix', required_shape=(3, 3))
        check_mtrx(t_vec, what_mtrx='3d translation vector', required_shape=(3,))

        rot_mat_4x4 = np.zeros((4, 4))
        rot_mat_4x4[:3, :3] = rot_mtrx
        rot_mat_4x4[3, 3] = 1

        t_vec_4d = np.hstack((t_vec, [0]))

        return RotoTranslate(rot_mat_4x4, t_vec_4d)

    
    
    @classmethod
    def from_vec_vec(cl, rot_vec, t_vec):
        rot_vec = np.array(rot_vec).astype(RotoTranslate.rt_transform_default_dtype) 
        t_vec = np.array(t_vec).astype(RotoTranslate.rt_transform_default_dtype)
        check_mtrx(rot_vec, what_mtrx='3d rotation vector', required_shape=(3,))
        check_mtrx(t_vec, what_mtrx='3d translation vector', required_shape=(3,))

        return RotoTranslate.from_mat_vec(cv2.Rodrigues(rot_vec)[0], t_vec)

    def __call__(self, pcd: PointCloud, inplace=False):
        if inplace:
            pcd._points = pcd.p4d
            np.dot(pcd.p4d, self._rot_mat.T, out=pcd._points)
            np.add(pcd._points,  self._t_vec.reshape((1, -1)), out=pcd._points)
        else:
            return PointCloud(pcd.p4d @ self._rot_mat.T + self._t_vec.reshape((1, -1)))

def test():
    pcd = PointCloud(np.random.random((10, 3)) * 10)
    print(pcd.p4d)
    rt_transform = RotoTranslate.from_vec_vec([0, 0, 0], [1, 1, 1])
    rt_transform(pcd, inplace=True)
    print(pcd.p3d)

    pcd2 = RotoTranslate.from_vec_vec([2, 5, 6], [1, 1, 1])(pcd)
    print(pcd2.p3d)

    print(pcd.p3d)

if __name__ == '__main__':
    test()


