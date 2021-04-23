from time import perf_counter
import numpy as np
import cv2 as cv2
import json as json
import matplotlib.pyplot as plt
import open3d as o3d

from kinect_telepresence.camera.intrinsics import DepthIntrinsics, RgbIntrinsics
from kinect_telepresence.camera.camera import Camera
from kinect_telepresence.geometry.homogeneous import RotoTranslate
from kinect_telepresence.geometry.pointcloud import PointCloud
from kinect_telepresence.distortion.undistort import DepthDistortionRectifier

def open3d_vis2(xyz, color, camera_intrinsic_params):
    color = color.astype(np.float32) / 255
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(xyz)
    pc.colors = o3d.utility.Vector3dVector(color.reshape(-1, 3))
    # o3d.io.write_point_cloud(fp, pc)
    pc.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    o3d.visualization.draw_geometries([pc])

def depth2xyd_list(depth):
    '''
    Create list of points in (x, y, depth) format from depth map.

    Parameters
    ----------
    depth: np.ndarray
    
    Returns
    -------
    np.ndarray
        Array with shape (3, n), n - number of points
    '''
    xx, yy = np.meshgrid(np.arange(depth.shape[1]), np.arange(depth.shape[0]))
    xyd_list = np.array([yy.reshape((-1)), xx.reshape((-1)), depth.reshape((-1))])
    return xyd_list[[1, 0, 2]]

def unproject_to_3d(xyd_list, intrinsic_mat):
    '''
    Get 3d coordinates of points from pixel coordinates and depth.

    {(x, y), depth_map} -> (X, Y, Z)

    Inverse projection procedure.

    Parameters
    ----------
    xyd_list: np.ndarray
        Array of points in (x, y, depth) format with shape (3, n).
    intrinsic_mat: np.ndarray
        Camera calibration matrix with shape (3, 3). 
    
    Returns
    -------
    np.ndarray
        Array with shape (3, n). 3d coordinates of points.

    '''
    p3d_x = (xyd_list[0] - intrinsic_mat[0, 2]) * xyd_list[2] / intrinsic_mat[0, 0]
    p3d_y = (xyd_list[1] - intrinsic_mat[1, 2]) * xyd_list[2] / intrinsic_mat[1, 1]
    return np.array([p3d_x, p3d_y, xyd_list[2]])


def main():
    path_input_depth = '../depth0592.png'
    path_input_color = '../color0592.png'

    json_path = 'mkv_meta.json'
    path_output_depth = 'depth_transformed.png'
    path_output_pcd = 'pc1.ply'

    with open(json_path) as f:
        params = json.load(f)
    
    depthmap = cv2.imread(path_input_depth, -1)
    rgb = cv2.imread(path_input_color)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)


    depth_intrinsics = DepthIntrinsics.from_meta_dict(params)
    rgb_intrinsics = RgbIntrinsics.from_meta_dict(params)

    t = np.array(params['depth_to_rgb']['t'])
    rot_vec = np.array(params['depth_to_rgb']['r'])

    rt_transform_to_depth  = RotoTranslate.from_vec_vec([0, 0, 0], [0, 0, 0])
    rt_transform_to_rgb = RotoTranslate.from_vec_vec(rot_vec, t)

    #===============
    start = perf_counter()

    rgb_camera = Camera(rgb_intrinsics, rt_transform_to_rgb)
    depth_camera = Camera(depth_intrinsics, rt_transform_to_depth)

    dist_rectifier = DepthDistortionRectifier(depth_intrinsics, 'bilinear')
    depthmap_undistorted = dist_rectifier(depthmap)
    pcd = depth_camera.depthmap2pcd_world(depthmap_undistorted)
    depth_reproj = rgb_camera.world_pcd2depthmap(pcd)
    
    end = perf_counter()
    print("elapsed time: ", end - start)
    #===============
    plt.imshow(depthmap)
    plt.show()
    plt.imshow(depthmap_undistorted)
    plt.show()


    plt.imshow(depth_reproj)
    plt.show()
    #===========
    color_undist = rgb.reshape((-1, 3))
    xyd_list = depth2xyd_list(depth_reproj).astype(np.float32)

    xyz3d_list = unproject_to_3d(xyd_list, rgb_intrinsics.cam_mtrx_undist).T

    print(">>", xyz3d_list.shape)
    print(">>", color_undist.shape)


    # color_undist = rgb[depth_reproj != 0].reshape((-1, 3))
    # xyz3d_list = rgb_camera._rt_transform_to_cam(pcd).p3d
    # print(xyz3d_list.shape, color_undist.shape)
    open3d_vis2(xyz3d_list, color_undist, {  'height': rgb_intrinsics.resolution[0],
                                            'width': rgb_intrinsics.resolution[1],
                                            'fx': rgb_intrinsics.cam_mtrx_undist[0, 0],
                                            'fy': rgb_intrinsics.cam_mtrx_undist[1, 1],
                                            'cx': rgb_intrinsics.cam_mtrx_undist[0, 2],
                                            'cy': rgb_intrinsics.cam_mtrx_undist[1, 2]})


    # pcd = depth_camera.depthmap2pcd_world(depthmap)

    # depth



if __name__ == '__main__':
    main()


