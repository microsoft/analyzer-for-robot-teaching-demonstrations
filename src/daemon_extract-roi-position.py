from pathlib import Path
import shutil
from time import time, time_ns
import zipfile
import cv2
import os.path as osp
import fastapi
import datetime
from fastapi import Form, UploadFile, File
from pathlib import Path
from tempfile import NamedTemporaryFile
from fastapi.responses import FileResponse
from pybsc import save_json
from fastapi_utils import save_upload_file_to_tmp
import cv2
import numpy as np
import os
import json
from utils.open3d_utils import create_pointcloud
from utils.open3d_utils import ExtractClusteredIndices
import open3d
import tempfile

__version__ = '0.0.1'

app = fastapi.FastAPI()

SERVICE = {
    "name": "position_extractor",
    "version": __version__,
    "libraries": {
        "position_extractor": __version__
    },
}


@app.get("/")
async def get_root():
    return {
        "service": SERVICE,
        "time": int(datetime.datetime.now().timestamp() * 1000),
    }


@app.get('/info', tags=['Utility'])
async def info():
    about = dict(
        version=__version__,
    )
    return about


@app.post("/position_extraction/")
def roi_based_extraction(upload_file_depth: UploadFile = File(None),
                         upload_file_rgb: UploadFile = File(None),
                         upload_json_roi: UploadFile = File(None),
                         upload_json_camera_param: UploadFile = File(None),
                         upload_json_contact_point: UploadFile = File(None),
                         color_filter: bool = Form(False),
                         depth_filter: bool = Form(True)):

    with tempfile.TemporaryDirectory() as output_dir:
        depth_file_path = save_upload_file_to_tmp(upload_file_depth)
        rgb_file_path = save_upload_file_to_tmp(upload_file_rgb)

        if upload_json_roi is not None:
            json_file_path = save_upload_file_to_tmp(upload_json_roi)
            with open(json_file_path) as json_file:
                data = json.load(json_file)
            x_min, y_min, x_max, y_max = data['left'], data['top'], data['right'], data['bottom']

        cameramodel = None
        if upload_json_camera_param is not None:
            camera_param_file_path = save_upload_file_to_tmp(
                upload_json_camera_param)
            with open(camera_param_file_path) as json_file:
                cameramodel = json.load(json_file)

        contact_point = None
        if upload_json_contact_point is not None:
            contact_point_file_path = save_upload_file_to_tmp(
                upload_json_contact_point)
            with open(contact_point_file_path) as json_file:
                data = json.load(json_file)
                # this is a relative position cropped from the original image
                x, y = data['x'], data['y']
                contact_point = (x_min + x, y_min + y)
            if os.path.exists(contact_point_file_path):
                Path(contact_point_file_path).unlink()
        print("=> Converting bgr images to rgb images")
        #depth_img = cv2.imread(str(depth_file_path), cv2.IMREAD_ANYDEPTH)
        depth_img = np.load(str(depth_file_path))
        depth_img = depth_img.astype(np.float32)
        rgb_img = None
        bgr_img = cv2.imread(str(rgb_file_path))
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        indices_image = np.arange(
            rgb_img.shape[0] * rgb_img.shape[1]).reshape(
                rgb_img.shape[0], rgb_img.shape[1])
        indices_image = np.array(indices_image, dtype=np.int32)
        print("=> Extracting hand positions")
        #import pdb; pdb.set_trace()
        open3d_cloud = create_pointcloud(
            rgb_img,
            depth_img[:, :, np.newaxis],
            cameramodel=cameramodel)
        roi_position = None
        ec = ExtractClusteredIndices(open3d_cloud)
        whole_points = np.array(open3d_cloud.points)
        whole_colors = np.array(open3d_cloud.colors)
        indices = indices_image[y_min:y_max, x_min:x_max]
        print("=> Debug1")
        if contact_point is not None:
            surrounding_pixels = 15
            x_min_contact = np.maximum(
                0, contact_point[0] - surrounding_pixels)
            x_max_contact = np.minimum(
                indices_image.shape[1],
                contact_point[0] + surrounding_pixels)
            y_min_contact = np.maximum(
                0, contact_point[1] - surrounding_pixels)
            y_max_contact = np.minimum(
                indices_image.shape[0],
                contact_point[1] + surrounding_pixels)
            indices_contact_point = indices_image[y_min_contact:y_max_contact, x_min_contact:x_max_contact]
        print("=> Debug2")
        cluster_indices, pcd = ec.filter_clustering(
            indices=indices.reshape(-1), min_points=100,
            voxel_size=0.01,
            eps=0.3)

        mode_cluster = False
        if len(cluster_indices) > 0 and mode_cluster:
            max_indices = sorted(
                cluster_indices,
                key=lambda x: len(x))[-1]
        else:
            max_indices = indices.reshape(-1)

        pcd_trim = open3d.geometry.PointCloud()
        pcd_trim.points = open3d.utility.Vector3dVector(
            whole_points[max_indices])
        pcd_trim.colors = open3d.utility.Vector3dVector(
            whole_colors[max_indices])
        pcd_trim.remove_non_finite_points()

        # same trimming for contact points
        cp_center = None
        if contact_point is not None:
            pcd_trim_cp = open3d.geometry.PointCloud()
            pcd_trim_cp.points = open3d.utility.Vector3dVector(
                whole_points[indices_contact_point.reshape(-1)])
            pcd_trim_cp.colors = open3d.utility.Vector3dVector(
                whole_colors[indices_contact_point.reshape(-1)])
            pcd_trim_cp.remove_non_finite_points()
            cp_xyz = np.array(pcd_trim_cp.points)

        if color_filter:
            print("=> Filtering colors")
            from sklearn.cluster import KMeans
            points_xyz = np.array(pcd_trim.points)
            colors_xyz = np.array(pcd_trim.colors)
            kmeans = KMeans(
                init="random",
                n_clusters=3,
                n_init=10,
                max_iter=300,
                random_state=42)
            kmeans.fit(colors_xyz)
            labels = kmeans.labels_
            unique, counts = np.unique(labels, return_counts=True)
            sel_color = labels == unique[np.argmax(counts)]
            pcd_trim_left = open3d.geometry.PointCloud()
            pcd_trim_left.points = open3d.utility.Vector3dVector(
                points_xyz[sel_color])
            pcd_trim_left.colors = open3d.utility.Vector3dVector(
                colors_xyz[sel_color])
            pcd_trim = pcd_trim_left
        if depth_filter:
            print("=> Filtering points")
            from sklearn.cluster import KMeans
            points_xyz = np.array(pcd_trim.points)
            colors_xyz = np.array(pcd_trim.colors)
            kmeans = KMeans(
                init="random",
                n_clusters=2,
                n_init=10,
                max_iter=300,
                random_state=42)
            kmeans.fit(points_xyz)
            labels = kmeans.labels_
            unique, counts = np.unique(labels, return_counts=True)

            # focus on the cluster closest to the camera
            z_list = []
            for i in range(len(unique)):
                sel_point = labels == unique[i]
                pcd_trim_tmp = open3d.geometry.PointCloud()
                pcd_trim_tmp.points = open3d.utility.Vector3dVector(
                    points_xyz[sel_point])
                pcd_trim_tmp.colors = open3d.utility.Vector3dVector(
                    colors_xyz[sel_point])
                roi_position_tmp = np.median(np.array(pcd_trim_tmp.points), axis=0)
                z_list.append(roi_position_tmp[2])

            # pick up the closest cluster
            sel_point = labels == unique[np.argmin(z_list)]
            # alternatively, pick up the largest cluster
            # sel_point = labels == unique[np.argmax(counts)]
            pcd_trim_left = open3d.geometry.PointCloud()
            pcd_trim_left.points = open3d.utility.Vector3dVector(
                points_xyz[sel_point])
            pcd_trim_left.colors = open3d.utility.Vector3dVector(
                colors_xyz[sel_point])

            pcd_trim = pcd_trim_left

            # statistic-based depth filter
            points_xyz = np.array(pcd_trim.points)
            colors_xyz = np.array(pcd_trim.colors)

            q1, q3 = np.percentile(points_xyz[:, 2], (25, 75))  # 25% and 75%
            iqr = q3 - q1
            outlier_threshold = q3 + 1.5 * iqr

            selected_indices = np.where(
                points_xyz[:, 2] < outlier_threshold)[0]
            points_xyz_filtered = points_xyz[selected_indices]
            pcd_trim_left = open3d.geometry.PointCloud()
            pcd_trim_left.points = open3d.utility.Vector3dVector(
                points_xyz_filtered)
            pcd_trim_left.colors = open3d.utility.Vector3dVector(
                colors_xyz[selected_indices])
            pcd_trim = pcd_trim_left

        # matching with contact point
        if contact_point is not None:
            pcd_trim_xyz = np.array(pcd_trim.points)
            is_found = np.zeros(len(cp_xyz), dtype=np.bool)
            for i in range(len(cp_xyz)):
                # check if the identical point is already in pcd_trim_xyz
                if np.any(np.isclose(pcd_trim_xyz, cp_xyz[i], atol=1e-08)):
                    is_found[i] = True
            cp_xyz = cp_xyz[is_found]
            if len(cp_xyz) > 0:
                cp_center = np.mean(cp_xyz, axis=0)

        roi_position = np.median(np.array(pcd_trim.points), axis=0)
        print(roi_position)

        out_filepaths = []
        open3d.io.write_point_cloud(output_dir + '/pcd_output.ply', pcd_trim)
        out_filepaths.append(output_dir + '/pcd_output.ply')
        pos_info = {'roi_position': list(roi_position)}
        info_json_path = osp.join(output_dir, 'position.json')
        save_json(pos_info, info_json_path)
        out_filepaths.append(info_json_path)

        if cp_center is not None:
            pos_info = {'contact_position': list(cp_center)}
            info_json_path = osp.join(output_dir, 'contact_position.json')
            save_json(pos_info, info_json_path)
            out_filepaths.append(info_json_path)

        zip_filename = str(Path(output_dir).with_suffix('.zip'))
        print(f'Saving to zip file {zip_filename}')
        zf = zipfile.ZipFile(zip_filename, "w")
        for fpath in out_filepaths:
            print(f'Saving {fpath} to zip file {zip_filename}')
            fdir, fname = os.path.split(fpath)
            zip_path = fname
            if os.path.exists(fpath):
                zf.write(fpath, zip_path)
                Path(fpath).unlink()
            else:
                import pdb
                pdb.set_trace()
        zf.close()
        # delete the tmp folders
        if os.path.exists(depth_file_path):
            Path(depth_file_path).unlink()
        if os.path.exists(rgb_file_path):
            Path(rgb_file_path).unlink()
        if os.path.exists(json_file_path):
            Path(json_file_path).unlink()
        if os.path.exists(camera_param_file_path):
            Path(camera_param_file_path).unlink()

        print('end extraction...')
        return FileResponse(
            zip_filename,
            media_type='application/x-zip-compressed',
            filename=f'position_extraction.zip')


@app.post("/plane_extraction/")
def convert_video(upload_file_depth: UploadFile = File(None),
                  upload_file_rgb: UploadFile = File(None),
                  upload_json_camera_param: UploadFile = File(None),
                  color_filter: bool = Form(True)):

    with tempfile.TemporaryDirectory() as output_dir:
        depth_file_path = save_upload_file_to_tmp(upload_file_depth)
        rgb_file_path = save_upload_file_to_tmp(upload_file_rgb)

        cameramodel = None
        if upload_json_camera_param is not None:
            camera_param_file_path = save_upload_file_to_tmp(
                upload_json_camera_param)
            with open(camera_param_file_path) as json_file:
                cameramodel = json.load(json_file)

        print("=> Converting bgr images to rgb images")
        depth_img = np.load(str(depth_file_path))
        depth_img = depth_img.astype(np.float32)
        rgb_img = None
        bgr_img = cv2.imread(str(rgb_file_path))
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        print("=> Extracting hand positions")
        open3d_cloud = create_pointcloud(
            rgb_img,
            depth_img[:, :, np.newaxis],
            cameramodel=cameramodel)
        open3d_cloud.remove_non_finite_points()
        plane_model, inliers = open3d_cloud.segment_plane(
            distance_threshold=0.01, ransac_n=3, num_iterations=1000)
        [a, b, c, d] = plane_model
        print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

        inlier_cloud = open3d_cloud.select_by_index(inliers)
        inlier_cloud.uniform_down_sample(10)

        if color_filter:
            print("=> Filtering colors")
            from sklearn.cluster import KMeans
            points_xyz = np.array(inlier_cloud.points)
            colors_xyz = np.array(inlier_cloud.colors)
            kmeans = KMeans(
                init="random",
                n_clusters=3,
                n_init=10,
                max_iter=300,
                random_state=42)
            kmeans.fit(colors_xyz)
            labels = kmeans.labels_
            unique, counts = np.unique(labels, return_counts=True)
            sel_color = labels == unique[np.argmax(counts)]
            pcd_trim_left = open3d.geometry.PointCloud()
            pcd_trim_left.points = open3d.utility.Vector3dVector(
                points_xyz[sel_color])
            pcd_trim_left.colors = open3d.utility.Vector3dVector(
                colors_xyz[sel_color])
            inlier_cloud = pcd_trim_left

        # save pcd
        open3d.io.write_point_cloud(
            output_dir + '/pcd_plane.ply', inlier_cloud)
        out_filepaths = [output_dir + '/pcd_plane.ply']

        zip_filename = str(Path(output_dir).with_suffix('.zip'))
        print(f'Saving to zip file {zip_filename}')
        zf = zipfile.ZipFile(zip_filename, "w")
        for fpath in out_filepaths:
            print(f'Saving {fpath} to zip file {zip_filename}')
            fdir, fname = os.path.split(fpath)
            zip_path = fname
            if os.path.exists(fpath):
                zf.write(fpath, zip_path)
                Path(fpath).unlink()
            else:
                import pdb
                pdb.set_trace()
        zf.close()
        # delete the tmp folders
        if os.path.exists(depth_file_path):
            Path(depth_file_path).unlink()
        if os.path.exists(rgb_file_path):
            Path(rgb_file_path).unlink()
        if os.path.exists(camera_param_file_path):
            Path(camera_param_file_path).unlink()
        return FileResponse(
            zip_filename,
            media_type='application/x-zip-compressed',
            filename=f'plane_extraction.zip')
