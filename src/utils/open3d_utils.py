import copy
import open3d
import numpy as np
import trimesh


def create_pointcloud(img, depth, cameramodel=None):
    rgbd_image = open3d.geometry.RGBDImage.create_from_color_and_depth(
        open3d.geometry.Image(img), open3d.geometry.Image(depth),
        depth_trunc=3.0,
        # depth_scale=1.0,
        convert_rgb_to_intensity=False)
    if cameramodel is not None:
        intrinsic = open3d.camera.PinholeCameraIntrinsic()
        intrinsic.set_intrinsics(
            cameramodel['width'],
            cameramodel['height'],
            cameramodel['fx'],
            cameramodel['fy'],
            cameramodel['cx'],
            cameramodel['cy'])
        print(intrinsic.intrinsic_matrix)
        print(intrinsic.width)
        print(intrinsic.height)
    else:
        intrinsic = open3d.camera.PinholeCameraIntrinsic(
            open3d.camera.PinholeCameraIntrinsicParameters.Kinect2DepthCameraDefault)
        # ptint
        print(intrinsic.intrinsic_matrix)
        print(intrinsic.width)
        print(intrinsic.height)
    pcd = open3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,
        intrinsic,
        project_valid_depth_only=False)
    return pcd


class ExtractClusteredIndices(object):

    def __init__(self, pcd):
        self.original_pcd = pcd
        self.non_nan_point = copy.deepcopy(pcd).remove_non_finite_points()

    def filter_clustering(self, indices=None,
                          voxel_size=0.03, min_points=40, eps=0.1):
        if indices is None:
            indices = np.arange(
                len(self.original_pcd.points))
            pcd = self.original_pcd
        else:
            pcd = open3d.geometry.PointCloud()
            pcd.points = open3d.utility.Vector3dVector(
                np.array(self.original_pcd.points)[
                    indices])
        min_bound = self.non_nan_point.get_min_bound() - \
            voxel_size * 0.5
        max_bound = self.non_nan_point.get_max_bound() + \
            voxel_size * 0.5
        voxel, _, original_point_indices = \
            pcd.voxel_down_sample_and_trace(
                voxel_size,
                min_bound, max_bound,
                approximate_class=True)
        labels = np.array(
            voxel.cluster_dbscan(
                eps=eps,
                min_points=min_points,
                print_progress=False))
        cluster_indices = []
        if len(labels) == 0:
            return cluster_indices, pcd
        max_label = labels.max()
        for label_index in range(max_label + 1):
            concat_indices = np.concatenate(
                [original_point_indices[index]
                 for index in np.where(labels == label_index)[0]])
            cluster_indices.append(indices[concat_indices])
        return cluster_indices, pcd


def visualize_pcd(pcd):
    non_nan_point = copy.deepcopy(pcd).remove_non_finite_points()
    points = np.array(non_nan_point.points)
    colors = np.array(non_nan_point.colors)
    pc = trimesh.PointCloud(
        vertices=points,
        colors=colors)
    pc.show()
