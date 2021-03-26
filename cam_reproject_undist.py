import numpy as np
import cv2 as cv2
import open3d as o3d
import matplotlib.pyplot as plt
import json
from undistort import undistort

# http://nicolas.burrus.name/index.php/Research/KinectCalibration
# https://stackoverflow.com/questions/31265245/extracting-3d-coordinates-given-2d-image-points-depth-map-and-camera-calibratio
# https://docs.opencv.org/3.4/da/d54/group__imgproc__transform.htm
# https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html

def intrinsic_mat(intrinsics, keys=['fx', 'fy', 'cx', 'cy']):
    '''
    Create calibration matrix from dict

            f_x    0    c_x
        K =  0    f_y   c_y
             0     0     1
    '''
    int_mat = np.array([[intrinsics[keys[0]],                0, intrinsics[keys[2]]],
                        [               0, intrinsics[keys[1]], intrinsics[keys[3]]],
                        [               0,                0,               1]])
    return int_mat


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


def drop_zero_depths(xyd_list):
    mask = xyd_list[2] != 0
    xyd_list = xyd_list[:, mask]
    return xyd_list


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


def undistort_points(xyd_list, intrinsic_mat, dist_coeffs, P):
    '''
    Perform distortion rectification of points.

    Parameters
    ----------
    xyd_list: np.ndarray
        Array of points in (x, y, depth) format with shape (3, n).
    intrinsic_mat: np.ndarray
        Camera calibration matrix with shape (3, 3). 
    dist_coeffs: np.ndarray
        Array with distortion coefficients with shape (8,).
        [k1, k2, p1, p2, k3, k4, k5, k6]

    Returns
    -------
    np.ndarray
        Array with shape (3, n)

    '''
    undist_res = np.zeros_like(xyd_list)
    undist_res[2] = xyd_list[2] # copy z-coord without changes
    print(intrinsic_mat)
    undist_res[:2] = cv2.undistortPoints(xyd_list[:2].copy(), intrinsic_mat, dist_coeffs, P=P)[:, 0, :].T

    # WTF? See: https://stackoverflow.com/questions/8499984/how-to-undistort-points-in-camera-shot-coordinates-and-obtain-corresponding-undi
    # undist_res[:2] = intrinsic_mat[:2, :2] @ undist_res[:2] + intrinsic_mat[:2, 2].reshape((-1, 1))
    
    return undist_res

def xyd_list2depthmap(xyd_list, depthmap_size):
    '''
    Make depth map from list of points
    '''
    new_depth = np.full(depthmap_size, np.inf)

    xyd_list[:2] = np.round(xyd_list[:2])
    mask = (xyd_list[0] >= 0) & (xyd_list[0] < depthmap_size[1]) & (xyd_list[1] >= 0) & (xyd_list[1] < depthmap_size[0])
    xyd_list = xyd_list[:, mask]
    
    for point in xyd_list.T:
        if new_depth[np.uint16(point[1]), np.uint16(point[0])] > point[2]:
            new_depth[np.uint16(point[1]), np.uint16(point[0])] = point[2]
    
    new_depth[new_depth == np.inf] = 0

    return new_depth

def reproject(depth, depth_intrinsic_mat, depth_intrinsic_undist_mat, rgb_intrinsic_mat, rgb_intrinsic_undist_mat, depth_dist_coeffs, rgb_dist_coeffs, rot_vec, t_vec, img_size):
    '''
    Perform reprojection of depth map.
    '''

    # depth = cv2.undistort(depth, depth_intrinsic_undist_mat, depth_dist_coeffs, None, None)

    # plt.imshow(depth)
    # plt.show()

    xyd_list = depth2xyd_list(depth).astype(np.float32)

    xyd_list = drop_zero_depths(xyd_list)
    
    xyd_list = undistort_points(xyd_list, depth_intrinsic_mat, depth_dist_coeffs)#, cv2.Rodrigues(rot_vec)[0], depth_intrinsic_undist_mat)
    
    xyz3d_list = unproject_to_3d(xyd_list, depth_intrinsic_mat)
    
    xy_list_reprojected = cv2.projectPoints(xyz3d_list, rot_vec, t_vec, rgb_intrinsic_undist_mat, np.array([]))[0][:, 0, :].T#, rgb_dist_coeffs)[0][:, 0, :].T
    
    xyd_list_reprojected = np.vstack((xy_list_reprojected, xyd_list[2]))

    new_depth = xyd_list2depthmap(xyd_list_reprojected, img_size)

    return new_depth

def reproject2(depth, depth_intrinsic_mat, depth_intrinsic_undist_mat, rgb_intrinsic_mat, rgb_intrinsic_undist_mat, depth_dist_coeffs, rgb_dist_coeffs, rot_vec, t_vec, img_size):
    '''
    Perform reprojection of depth map.
    '''

    # depth = cv2.undistort(depth, depth_intrinsic_undist_mat, depth_dist_coeffs, None, None)
    depth = undistort(depth, depth_intrinsic_mat, depth_intrinsic_undist_mat, depth_dist_coeffs)
    # plt.imshow(depth)
    # plt.show()

    xyd_list = depth2xyd_list(depth.copy()).astype(np.float32)
    # print("!", xyd_list.shape)
    # xyd_list = drop_zero_depths(xyd_list)
    
    # xyd_list = undistort_points(xyd_list, depth_intrinsic_mat, depth_dist_coeffs, depth_intrinsic_undist_mat)#, cv2.Rodrigues(rot_vec)[0], depth_intrinsic_undist_mat)
    
    # after = xyd_list2depthmap(xyd_list.copy(), depth.shape)
    # print("!", after.shape)
    # plt.subplot(131)
    # plt.title("after")
    # plt.imshow(after)
    # plt.subplot(132)
    # plt.title("before")
    # plt.imshow(depth)
    # plt.subplot(133)
    # diff = depth - after
    # print("LOL", np.amax(diff), np.amin(diff))
    # plt.imshow(diff / np.amax(diff))
    # plt.show()
    # cv2.imwrite("depth_undist.png",after)
    xyz3d_list = unproject_to_3d(xyd_list, depth_intrinsic_undist_mat)
    
    xyz3d_list = cv2.Rodrigues(rot_vec)[0] @ xyz3d_list + t_vec.reshape((-1, 1))
    
    
    xyz3d_list = rgb_intrinsic_undist_mat @ xyz3d_list
    
    # xyz3d_list = undistort_points(xyz3d_list, depth_intrinsic_mat, depth_dist_coeffs, depth_intrinsic_undist_mat) #<<
    
    

    xy_list_reprojected = np.array([xyz3d_list[0] / xyz3d_list[2], xyz3d_list[1] / xyz3d_list[2]])

    # xy_list_reprojected = cv2.projectPoints(xyz3d_list, rot_vec, t_vec, rgb_intrinsic_undist_mat, np.array([]))[0][:, 0, :].T#, rgb_dist_coeffs)[0][:, 0, :].T
    
    xyd_list_reprojected = np.vstack((xy_list_reprojected, xyz3d_list[2]))
    print("!!!", xyd_list_reprojected.shape)
    new_depth = xyd_list2depthmap(xyd_list_reprojected, img_size)
    print("!!!", new_depth.shape)
    return new_depth#xyz3d_list


def open3d_vis(depth, color, camera_intrinsic_params):
    camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(**camera_intrinsic_params)

    color_raw = o3d.geometry.Image(color.astype(np.uint8))
    depth_raw = o3d.geometry.Image(depth.astype(np.uint16))
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw, depth_scale=1000, convert_rgb_to_intensity=False, depth_trunc=10.0)

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, camera_intrinsic)

    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    o3d.visualization.draw_geometries([pcd])

def open3d_vis2(xyz, color, camera_intrinsic_params):
    color = color.astype(np.float32) / 255
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(xyz)
    pc.colors = o3d.utility.Vector3dVector(color.reshape(-1, 3))
    # o3d.io.write_point_cloud(fp, pc)
    pc.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    o3d.visualization.draw_geometries([pc])

    o3d.io.write_point_cloud("pc1.ply", pc)
    # camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(**camera_intrinsic_params)

    # color_raw = o3d.geometry.Image(color.astype(np.uint8))
    # depth_raw = o3d.geometry.Image(depth.astype(np.uint16))
    # rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw, depth_scale=1000, convert_rgb_to_intensity=False, depth_trunc=10.0)

    # pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, camera_intrinsic)

    # pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    # o3d.visualization.draw_geometries([pcd])

def main():
    # path_input_depth = '001236.png'
    # path_input_color = '001236.jpg'
    path_input_depth = 'depth0177.png'
    path_input_color = 'color0177.png'
    
    json_path = 'mkv_meta.json'
    path_output_depth = 'depth_transformed.png'


    depth = cv2.imread(path_input_depth, -1)
    color = cv2.imread(path_input_color)
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)

    with open(json_path) as f:
        params = json.load(f)
    depth_intrinsics = params['depth_intrinsics']
    rgb_intrinsics = params['rgb_intrinsics']

    depth_intrinsic_undist_mat = intrinsic_mat(params['depth_undistorted_intrinsics'], keys=['fx', 'fy', 'px', 'py'])
    rgb_intrinsic_undist_mat = intrinsic_mat(params['rgb_undistorted_intrinsics'])


    depth_intrinsic_mat = intrinsic_mat(depth_intrinsics)
    depth_dist_coeffs = np.array([depth_intrinsics["k1"],
                                depth_intrinsics["k2"],
                                depth_intrinsics["p1"],
                                depth_intrinsics["p2"],
                                depth_intrinsics["k3"],
                                depth_intrinsics["k4"],
                                depth_intrinsics["k5"],
                                depth_intrinsics["k6"]
                            ])

    rgb_intrinsic_mat = intrinsic_mat(rgb_intrinsics)
    rgb_dist_coeffs = np.array([rgb_intrinsics["k1"],
                                rgb_intrinsics["k2"],
                                rgb_intrinsics["p1"],
                                rgb_intrinsics["p2"],
                                rgb_intrinsics["k3"],
                                rgb_intrinsics["k4"],
                                rgb_intrinsics["k5"],
                                rgb_intrinsics["k6"]
                            ])

    plt.imshow(color)
    plt.show()
    color_undist = color.copy()#cv2.undistort(color.copy(), rgb_intrinsic_mat, rgb_dist_coeffs,  newCameraMatrix=rgb_intrinsic_undist_mat)
    # color_undist = cv2.undistort(color_undist, rgb_intrinsic_mat, rgb_dist_coeffs, newCameraMatrix=rgb_intrinsic_undist_mat)
    plt.imshow(color_undist)
    plt.show()
    cv2.imwrite("color_undist2.png", color_undist[:, :, [2, 1, 0]])


    t = np.array(params['depth_to_rgb']['t'])
    rot_vec = np.array(params['depth_to_rgb']['r'])
    dst_size = (1536, 2048)

    # depth_reprojected = reproject2(depth, 
    #                             depth_intrinsic_mat, 
    #                             depth_intrinsic_undist_mat, 
    #                             rgb_intrinsic_mat, 
    #                             rgb_intrinsic_undist_mat, 
    #                             depth_dist_coeffs, 
    #                             rgb_dist_coeffs, 
    #                             rot_vec, t, dst_size)

    depth_reprojected_3d = reproject2(depth, 
                                depth_intrinsic_mat, 
                                depth_intrinsic_undist_mat, 
                                rgb_intrinsic_mat, 
                                rgb_intrinsic_undist_mat, 
                                depth_dist_coeffs, 
                                rgb_dist_coeffs, 
                                rot_vec, t, dst_size)


    plt.figure(figsize=(24,16))
    plt.subplot(131)
    plt.imshow(depth)
    plt.title('Original depth map')

    plt.subplot(132)
    plt.imshow(color_undist)
    plt.title('RGB image (undist)')

    plt.subplot(133)
    plt.imshow(depth_reprojected_3d)
    plt.title('Reprojected undistorted depth map')
    plt.show()

    color_undist = color_undist.reshape((-1, 3))


    # color = color_undist.reshape((-1, 3))[depth_reprojected_3d.reshape((-1, 1)) > 0]

    # open3d_vis(depth_reprojected, color_undist, {  'height': params['color_resolution']['h'],
    #                                         'width': params['color_resolution']['w'],
    #                                         'fx': params['rgb_undistorted_intrinsics']['fx'],
    #                                         'fy': params['rgb_undistorted_intrinsics']['fy'],
    #                                         'cx': params['rgb_undistorted_intrinsics']['cx'],
    #                                         'cy': params['rgb_undistorted_intrinsics']['cy']})
    print("!!!!", depth_reprojected_3d[depth_reprojected_3d != 0].shape, depth_reprojected_3d.shape)

    xyd_list = depth2xyd_list(depth_reprojected_3d).astype(np.float32)

    xyz3d_list = unproject_to_3d(xyd_list, rgb_intrinsic_undist_mat).T

    print(">>", xyz3d_list.shape)
    print(">>", color_undist.shape)

    open3d_vis2(xyz3d_list, color_undist, {  'height': params['color_resolution']['h'],
                                            'width': params['color_resolution']['w'],
                                            'fx': params['rgb_undistorted_intrinsics']['fx'],
                                            'fy': params['rgb_undistorted_intrinsics']['fy'],
                                            'cx': params['rgb_undistorted_intrinsics']['cx'],
                                            'cy': params['rgb_undistorted_intrinsics']['cy']})

    cv2.imwrite(path_output_depth, np.uint16(depth_reprojected_3d))

if __name__ == '__main__':
    main()
