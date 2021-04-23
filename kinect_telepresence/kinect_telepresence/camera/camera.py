import numpy as np
import cv2 as cv2
from kinect_telepresence.camera.intrinsics import Intrinsics
from kinect_telepresence.geometry.pointcloud import PointCloud
from kinect_telepresence.geometry.homogeneous import RotoTranslate

class Camera:
    def __init__(self, intrinsics: Intrinsics, rt_transform_to_cam: RotoTranslate):
        self._intrinsics = intrinsics
        self._rt_transform_to_cam = rt_transform_to_cam
    
    def project_to_cam_frame(self, pcd: PointCloud):
        '''
        Project point cloud using camera intrinsics into local camera frame.

        
        Returns list of triplets (u, v, d), where u and v are coordinates of points in camera local frame,
        d - depth of point (taken unchanged from original point cloud).

        Note: Intrinsics class implies 3x3 projection matrix (change??), so Pointcloud class
        is used as list of 3d points (i.e. using property function .p3d).

        Note: camera matrix for undistorted points is used here.
        '''
        proj_points = self._rt_transform_to_cam(pcd).p3d @ self._intrinsics.cam_mtrx_undist.T 
        proj_points[:, :2] = proj_points[:, :2] / proj_points[:, 2:]
        return PointCloud(proj_points)

    def world_pcd2depthmap(self, pcd: PointCloud):
        '''
        Project point cloud to camera frame and compose a depthmap.

        This function is the composition of Camera.project_to_cam_frame() followed by PointCloud.compose_depth().

        Parameters
        ----------
        pcd : kinect_telepresence.geometry.pointcloud.PointCloud
            Point cloud in a world frame
        
        Return
        ------
            np.ndarray


        '''
        return self.project_to_cam_frame(pcd).compose_depth(self._intrinsics.resolution)
        

    def unproject_to_world(self, pcd):
        '''
        Unproject points in camera frame to world coordinates.

        Parameters
        ----------
        pcd: kinect_telepresence.geometry.pointcloud.PointCloud
            List of triplets (u, v, d), where u, v are coordinates in local camera frame, d - depth in local camera frame

        Return
        ------
            kinect_telepresence.geometry.pointcloud.PointCloud

        '''
        xyd_list = pcd.p3d.T
        p3d_x = (xyd_list[0] - self._intrinsics.cam_mtrx_undist[0, 2]) * xyd_list[2] / self._intrinsics.cam_mtrx_undist[0, 0]
        p3d_y = (xyd_list[1] - self._intrinsics.cam_mtrx_undist[1, 2]) * xyd_list[2] / self._intrinsics.cam_mtrx_undist[1, 1]
        pcd_world = PointCloud(np.array([p3d_x, p3d_y, xyd_list[2]]).T)
        return self._rt_transform_to_cam.inv(pcd_world)
        

    def depthmap2pcd_world(self, depthmap: np.ndarray):
        '''
        '''
        return self.unproject_to_world(PointCloud.from_depth(depthmap))

