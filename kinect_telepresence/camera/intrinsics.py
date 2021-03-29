'''
class Intrinsics
Stores intrinsic camera parameters, such as camera matrix and distortion coefficients
'''

from utils.utils import check_mtrx
import numpy as np


class Intrinsics:
    def __init__(self, cam_mtrx_dist: np.ndarray, 
                cam_mtrx_undist: np.ndarray = None,
                resolution = None, 
                dist_coeffs = None, 
                other_params: dict = None):

        self._cam_mtrx_dist = np.array(cam_mtrx_dist)
        check_mtrx(self._cam_mtrx_dist, 'intrinsic matrix')

        self._cam_mtrx_undist = cam_mtrx_undist
        if not self._cam_mtrx_undist is None:
            self._cam_mtrx_undist = np.array(self._cam_mtrx_undist)
            check_mtrx(self._cam_mtrx_undist, 'intrinsic undistorted matrix', required_shape = (3, 3))

        self._resolution = resolution
        if not self._resolution is None:
            self._resolution = np.array(self._resolution)
            check_mtrx(self._resolution, 'resolution', required_shape = (2,))

        self._dist_coeffs = dist_coeffs
        if not self._dist_coeffs is None:
            self._dist_coeffs = np.array(self._dist_coeffs)
            check_mtrx(self._dist_coeffs, 'distortion coefficients list', required_shape = (8,))
        
        self._other_params = other_params
        if not self._other_params is None:
            if type(self._other_params) != dict:
                raise ValueError("other_params should be a dict")


    @property
    def cam_mtrx_dist(self):
        return self._cam_mtrx_dist

    @property
    def cam_mtrx_undist(self):
        if self._cam_mtrx_undist is None:
            raise RuntimeError("Camera undistorted matrix is not specified. Run calculate_undist_matrix() before call.")
        return self._cam_mtrx_undist
    
    @property
    def resolution(self):
        if self._resolution is None:
            raise RuntimeError("Resolution is not specified.")
        return tuple(self._resolution)

    @property
    def dist_coeffs(self):
        if self._dist_coeffs is None:
            raise RuntimeError("Distortion coefficients are not specified")
        return tuple(self._dist_coeffs)
    
    def __getattr__(self, name):
        if self._other_params is None:
            raise AttributeError(f"Undefined attribute {name}. _other_params dict is empty")
        elif not name in self._other_params.keys():
            raise AttributeError(f"Undefined attribute {name}. Not present at _other_params dict")
        else:
            return self._other_params[name]
    

    @classmethod
    def from_meta_dict(cls, meta_dict: dict):
        raise NotImplementedError

    def get_meta_dict(self):
        raise NotImplementedError

    def calculate_undist_matrix(self):
        raise NotImplementedError

def test():
    a = Intrinsics(np.eye(3), other_params={'keka':42})
    print(a.keka)

if __name__ == '__main__':
    test()