'''
class Pointcloud
'''

import numpy as np
from kinect_telepresence.camera.intrinsics import Intrinsics
from kinect_telepresence.utils.utils import check_mtrx
import open3d as o3d

class PointCloud:

    pcd_default_dtype = np.float32
    
    def __init__(self, points):
        self._points = np.array(points).astype(PointCloud.pcd_default_dtype)
        if len(self._points.shape) != 2:
            raise ValueError("Wrong points array shape. It should have (N, 3) or (N, 4) size.")
        if not self._points.shape[1] in (3, 4):
            raise ValueError("Wrong points array shape. It should have (N, 3) or (N, 4) size.")

    def __len__(self):
        return self._points.shape[0]

    @classmethod   
    def from_depth(cl, depthmap: np.ndarray):
        '''
        Do not confuse with unprojection of depthmap, acquired by certain depth camera, to 3d space.
        
        This function returns pointcloud with coordinates (i, j, d), where i - number of row, 
        j - number of column, d - depth stored in pixel (i,j).
        '''
        depthmap = np.array(depthmap)
        if len(depthmap.shape) != 2:
            raise ValueError("Wrong depth shape. Only 1-channel depthmaps are allowed")
        xx, yy = np.meshgrid(np.arange(depthmap.shape[1]), np.arange(depthmap.shape[0]))
        points = np.array([yy.reshape((-1)), xx.reshape((-1)), depthmap.reshape((-1))])
        points = points[[1, 0, 2]]
        return PointCloud(points.T)

    def compose_depth(self, depthmap_size):
        '''
        Create a depth map using points of point cloud.
        
        Treat point cloud as list of triplets (i, j, d), where i (rounded to the closest int) - number of row, 
        j (rounded to the closest int) - number of column, d - depth stored in pixel (i,j).

        For two points with coinciding i's and j's stores one with the smallest value of d (i.e. take the closest point).

        Note: do not confuse with projection of pointcloud using certain camera.
        '''
        check_mtrx(np.array(depthmap_size), "depth map size", (2,))

        depthmap = np.zeros(depthmap_size)

        xyd_list = self.p3d.T

        xyd_list[:2] = np.round(xyd_list[:2])

        mask = (xyd_list[0] >= 0) & (xyd_list[0] < depthmap_size[1]) & (xyd_list[1] >= 0) & (xyd_list[1] < depthmap_size[0])
        xyd_list = xyd_list[:, mask]
        
        depthmap[np.uint16(xyd_list[1]), np.uint16(xyd_list[0])] = xyd_list[2]


        return depthmap

    @property
    def p3d(self):
        '''
        Get list of 3d inhomogenious points coordinates.
        '''
        if self._points.shape[1] == 3:
            return self._points
        else:
            return self._points[:, :3] / self._points[:, 3:]

    @property
    def p4d(self):
        '''
        Get list of 4d homogenious points coordinates.
        '''
        if self._points.shape[1] == 4:
            return self._points
        else:
            return np.hstack((self._points, np.ones((self._points.shape[0], 1))))

    def open3d_visualize(self, color_image=None):
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(self.p3d)
        if not color_image is None:
            if color_image.reshape(-1, 3).shape != self.p3d.shape:
                raise ValueError("Number of points in point cloud and color image don't match")
            pc.colors = o3d.utility.Vector3dVector(color_image.reshape(-1, 3))
        pc.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

        o3d.visualization.draw_geometries([pc])

        return pc

        

    

def test():
    pcd = PointCloud([[1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9],
                    [10, 11, 12]])
    print(pcd.p3d)
    print(pcd.p4d)

    pcd = PointCloud([[1, 2, 3, 2],
                    [4, 5, 6, 6],
                    [7, 8, 9, 1],
                    [10, 11, 12, 10]])
    print(pcd.p3d)
    print(pcd.p4d)
    print(len(pcd))

def test2():
    depth = [[1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]]
    pcd = PointCloud.from_depth(depth, )
    print(pcd.p3d)
    print(len(pcd))

if __name__ == '__main__':
    test2()