import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import io
import cv2
import json

# img = cv2.imread('/home/kvsoshin/Kinect Telepresence/depth0592.png', -1)
img = cv2.imread('output_aligned.png', -1)
# img = cv2.imread('/home/kvsoshin/Kinect Telepresence/output_aligned6.png', -1)
color = cv2.imread('color0177.png', -1).astype(np.uint16)
color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)

# color_raw = o3d.io.read_image("/home/kvsoshin/Kinect Telepresence/output_aligned2.png")
color_raw = o3d.geometry.Image(color.astype(np.uint8))
# depth_raw = o3d.io.read_image("/home/kvsoshin/Kinect Telepresence/color0592.png")
depth_raw = o3d.geometry.Image(img.astype(np.uint16))
print(color_raw)
print(depth_raw)
rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw, depth_scale=1000, convert_rgb_to_intensity=False, depth_trunc=10.0)



with open('mkv_meta.json') as f:
    intrinsic_params = json.load(f)
print(np.amax(img), np.amin(img))
print(o3d.__version__)
camera_intrinsic_params = {}
tmp = intrinsic_params['rgb_undistorted_intrinsics'].copy()
tmp.update(intrinsic_params['color_resolution'].copy())
# tmp = intrinsic_params['depth_intrinsics'].copy()
# tmp.update(intrinsic_params['depth_resolution'].copy())
camera_intrinsic_params['height'] = tmp['h']
camera_intrinsic_params['width'] = tmp['w']
camera_intrinsic_params['fx'] = tmp['fx']
camera_intrinsic_params['fy'] = tmp['fy']
camera_intrinsic_params['cx'] = tmp['cx']
camera_intrinsic_params['cy'] = tmp['cy']

print(camera_intrinsic_params)

# o3d_depth = o3d.geometry.Image(img)
# camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
#     o3d.camera.PinholeCameraIntrinsicParameters.Kinect2ColorCameraDefault)
# camera_intrinsic = o3d.io.read_pinhole_camera_intrinsic('data/mkv_meta.json')
camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(**camera_intrinsic_params)



# pcd = o3d.geometry.PointCloud.create_from_depth_image(o3d_depth, camera_intrinsic)
# rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color, img)
pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, camera_intrinsic)



o3d.io.write_point_cloud('depth.ply', pcd, print_progress=True)

pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

o3d.visualization.draw_geometries([pcd])

plt.imshow(np.array(rgbd_image.depth))
plt.show()
