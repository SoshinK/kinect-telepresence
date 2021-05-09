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
from kinect_telepresence.distortion.undistort import DepthDistortionRectifier, RgbDistortionRectifier
from kinect_telepresence.filtering.jbu import JBU

from pathlib import Path

SCRIPT_PATH = Path(__file__).parent.absolute()
TRUNK_PATH = SCRIPT_PATH.parent
print(SCRIPT_PATH)

def main():
    path_input_depth = '../depth0592.png'
    path_input_color = '../color0592.png'

    json_path = 'mkv_meta.json'
    path_output_depth = 'depth_transformed.png'
    path_output_pcd = 'pc1.ply'

    with open(json_path) as f:
        params = json.load(f)
    depth_intrinsics = DepthIntrinsics.from_meta_dict(params)   
    rgb_intrinsics = RgbIntrinsics.from_meta_dict(params)


    depth_dist_rectifier = DepthDistortionRectifier(depth_intrinsics, 'bilinear')
    rgb_dist_rectifier = RgbDistortionRectifier(rgb_intrinsics, 'bilinear')
    
    with open(path_input_depth) as f:
        depthmap = cv2.imread(path_input_depth, -1)
    with open(path_input_color) as f:
        rgb = cv2.cvtColor(cv2.imread(path_input_color), cv2.COLOR_BGR2RGB)
        # rgb = rgb_dist_rectifier(rgb)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


    t = np.array(params['depth_to_rgb']['t'])
    rot_vec = np.array(params['depth_to_rgb']['r'])

    rt_transform_identical  = RotoTranslate.from_vec_vec([0, 0, 0], [0, 0, 0])
    rt_transform_to_rgb = RotoTranslate.from_vec_vec(rot_vec, t)

    rgb_camera = Camera(rgb_intrinsics, rt_transform_to_rgb)
    depth_camera = Camera(depth_intrinsics, rt_transform_identical)
    

    jbu = JBU(
        2, 
        1.5, 
        sigma_range=np.std(gray), 
        scale=1.)

    #===============
    depthmap_undistorted = depth_dist_rectifier(depthmap)

    pcd = depth_camera.depthmap2pcd_world(depthmap_undistorted)
    depth_reproj = rgb_camera.world_pcd2depthmap(pcd)
    
    start = perf_counter()

    depth_upsampled = jbu(depth_reproj, gray)

    print("Joint bilateral upsampling: ", perf_counter() - start)
    
    # plt.imshow(depth_upsampled)
    # plt.show()

    pcd_usampled = Camera(rgb_intrinsics, rt_transform_identical).depthmap2pcd_world(depth_upsampled)
    open3d_pcd = pcd_usampled.open3d_visualize(rgb.astype(np.float32) / 255.)

    cv2.imwrite(path_output_depth, depth_upsampled)
    o3d.io.write_point_cloud(path_output_pcd, open3d_pcd)



if __name__ == '__main__':
    main()


