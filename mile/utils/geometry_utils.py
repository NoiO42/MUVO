import numpy as np
import torch
from mile.data.dataset_utils import preprocess_gps


def bev_params_to_intrinsics(size, scale, offsetx):
    """
        size: number of pixels (width, height)
        scale: pixel size (in meters)
        offsetx: offset in x direction (direction of car travel)
    """
    intrinsics_bev = np.array([
        [1/scale, 0, size[0]/2 + offsetx],
        [0, -1/scale, size[1]/2],
        [0, 0, 1]
    ], dtype=np.float32)
    return intrinsics_bev


def intrinsics_inverse(intrinsics):
    fx = intrinsics[..., 0, 0]
    fy = intrinsics[..., 1, 1]
    cx = intrinsics[..., 0, 2]
    cy = intrinsics[..., 1, 2]
    one = torch.ones_like(fx)
    zero = torch.zeros_like(fx)
    intrinsics_inv = torch.stack((
        torch.stack((1/fx, zero, -cx/fx), -1),
        torch.stack((zero, 1/fy, -cy/fy), -1),
        torch.stack((zero, zero, one), -1),
    ), -2)
    return intrinsics_inv


def get_out_of_view_mask(cfg):
    """ Returns a mask of everything that is not visible from the image given a certain bird's-eye view grid."""
    fov = cfg.IMAGE.FOV
    w = cfg.IMAGE.SIZE[1]
    resolution = cfg.BEV.RESOLUTION

    f = w / (2 * np.tan(fov * np.pi / 360.0))
    c_u = w / 2 - cfg.IMAGE.CROP[0]  # Adjust center point due to cropping

    bev_left = -np.round((cfg.BEV.SIZE[0] // 2) * resolution, decimals=1)
    bev_right = np.round((cfg.BEV.SIZE[0] // 2) * resolution, decimals=1)
    bev_bottom = 0.01
    # The camera is not exactly at the bottom of the bev image, so need to offset it.
    camera_offset = (cfg.BEV.SIZE[1] / 2 + cfg.BEV.OFFSET_FORWARD) * resolution + cfg.IMAGE.CAMERA_POSITION[0]
    bev_top = np.round(cfg.BEV.SIZE[1] * resolution - camera_offset, decimals=1)

    x, z = np.arange(bev_left, bev_right, resolution), np.arange(bev_bottom, bev_top, resolution)
    ucoords = x / z[:, None] * f + c_u

    # Return all points which lie within the camera bounds
    new_w = cfg.IMAGE.CROP[2] - cfg.IMAGE.CROP[0]
    mask = (ucoords >= 0) & (ucoords < new_w)
    mask = ~mask[::-1]
    mask_behind_ego_vehicle = np.ones((int(camera_offset / resolution), mask.shape[1]), dtype=np.bool)
    return np.vstack([mask, mask_behind_ego_vehicle])


def calculate_geometry(image_fov, height, width, forward, right, up, pitch, yaw, roll):
    """Intrinsics and extrinsics for a single camera.
    See https://github.com/bradyz/carla_utils_fork/blob/dynamic-scene/carla_utils/leaderboard/camera.py
    and https://github.com/bradyz/carla_utils_fork/blob/dynamic-scene/carla_utils/recording/sensors/camera.py
    """
    f = width / (2 * np.tan(image_fov * np.pi / 360.0))
    cx = width / 2
    cy = height / 2
    intrinsics = np.float32([[f, 0, cx], [0, f, cy], [0, 0, 1]])
    extrinsics = get_extrinsics(forward, right, up, pitch, yaw, roll)
    return intrinsics, extrinsics


def get_extrinsics(forward, right, up, pitch, yaw, roll):
    # After multiplying the image coordinates by in the inverse intrinsics,
    # the resulting coordinates are defined with the axes (right, down, forward)
    assert pitch == yaw == roll == 0.0

    # After multiplying by the extrinsics, we want the axis to be (forward, left, up), and centered in the
    # inertial center of the ego-vehicle.
    mat = np.float32([
        [0,  0,  1, forward],
        [-1, 0,  0, -right],
        [0,  -1, 0, up],
        [0,  0,  0, 1],
    ])

    return mat


def lidar_to_histogram_features(lidar, cfg, crop=256):
    """
    Convert LiDAR point cloud into 2-bin histogram over 256x256 grid
    """

    # fit the center of the histogram the same as the bev.
    offset = np.asarray(cfg.VOXEL.EV_POSITION) * cfg.VOXEL.RESOLUTION  # ego position relative to min boundary.
    pixels_per_meter = cfg.POINTS.HISTOGRAM.RESOLUTION
    hist_max_per_pixel = cfg.POINTS.HISTOGRAM.HIST_MAX
    x_range = cfg.POINTS.HISTOGRAM.X_RANGE
    y_range = cfg.POINTS.HISTOGRAM.Y_RANGE
    z_range = cfg.POINTS.HISTOGRAM.Z_RANGE

    # 256 x 256 grid
    xbins = np.linspace(
        -offset[0],
        -offset[0] + x_range / pixels_per_meter,
        x_range + 1
    )
    ybins = np.linspace(
        -offset[1],
        -offset[1] + y_range / pixels_per_meter,
        y_range + 1,
        )
    zbins = np.linspace(
        -offset[2],
        -offset[2] + z_range / pixels_per_meter,
        z_range + 1
    )

    def splat_points(point_cloud, bins1, bins2):
        hist = np.histogramdd(point_cloud, bins=(bins1, bins2))[0]
        hist[hist > hist_max_per_pixel] = hist_max_per_pixel
        overhead_splat = hist / hist_max_per_pixel
        # return overhead_splat[::-1, ::-1]
        return overhead_splat

    # xy plane
    below = lidar[lidar[..., 2] <= 0][..., :2]
    middle = lidar[(0 < lidar[..., 2]) & (lidar[..., 2] <= 2.5)][..., :2]
    above = lidar[lidar[..., 2] > 2.5][..., :2]
    below_features = splat_points(below, xbins, ybins)
    middle_features = splat_points(middle, xbins, ybins)
    above_features = splat_points(above, xbins, ybins)
    total_features_xy = below_features + middle_features + above_features
    features_xy = np.stack([below_features, middle_features, above_features, total_features_xy], axis=-1)
    features_xy = np.transpose(features_xy, (2, 0, 1)).astype(np.float32)

    # xz plane
    left = lidar[lidar[..., 1] >= 1.5][..., ::2]
    center = lidar[(-1.5 < lidar[..., 1]) & (lidar[..., 1] < 1.5)][..., ::2]
    right = lidar[lidar[..., 1] <= -1.5][..., ::2]
    left_features = splat_points(left, xbins, zbins)
    center_features = splat_points(center, xbins, zbins)
    right_features = splat_points(right, xbins, zbins)
    total_features_xz = left_features + center_features + right_features
    features_xz = np.stack([left_features, center_features, right_features, total_features_xz], axis=-1)
    features_xz = np.transpose(features_xz, (2, 0, 1)).astype(np.float32)

    # yz plane
    behind = lidar[lidar[..., 0] < -2.5][..., 1:]
    mid = lidar[(-2.5 <= lidar[..., 0]) & (lidar[..., 0] <= 10)][..., 1:]
    front = lidar[lidar[..., 0] > 10][..., 1:]
    behind_features = splat_points(behind, ybins, zbins)
    mid_features = splat_points(mid, ybins, zbins)
    front_features = splat_points(front, ybins, zbins)
    total_features_yz = behind_features + mid_features + front_features
    features_yz = np.stack([behind_features, mid_features, front_features, total_features_yz], axis=-1)
    features_yz = np.transpose(features_yz, (2, 0, 1)).astype(np.float32)
    return features_xy, features_xz, features_yz


class PointCloud(object):
    def __init__(self, H=64, W=1024, fov_down=-30, fov_up=10, lidar_position=(1, 0, 2)):
        self.fov_up = fov_up / 180.0 * np.pi  # in rad
        self.fov_down = fov_down / 180.0 * np.pi
        self.fov = self.fov_up - self.fov_down
        self.H = H
        self.W = W
        self.lidar_position = np.asarray(lidar_position)

    def do_range_projection(self, points, semantics):
        points_carla = points * np.array([1, -1, 1])
        points_carla -= self.lidar_position

        depth = np.linalg.norm(points_carla, 2, axis=1)

        x = points_carla[:, 0]
        y = -points_carla[:, 1]  # carla-coor is left-hand.
        z = points_carla[:, 2]

        yaw = np.arctan2(y, x)
        pitch = np.arcsin(z / depth)

        proj_w = 0.5 * (1.0 - yaw / np.pi)
        proj_h = 1.0 - (pitch + abs(self.fov_down)) / self.fov
        proj_w *= self.W
        proj_h *= self.H

        proj_w = np.floor(proj_w)
        proj_w = np.minimum(self.W - 1, proj_w)
        proj_w = np.maximum(0, proj_w).astype(np.int32)

        proj_h = np.floor(proj_h)
        proj_h = np.minimum(self.H - 1, proj_h)
        proj_h = np.maximum(0, proj_h).astype(np.int32)

        order = np.argsort(depth)[::-1]
        depth = depth[order]
        proj_w = proj_w[order]
        proj_h = proj_h[order]
        points = points[order]
        semantics = semantics[order]

        range_depth = np.full((self.H, self.W), -1, dtype=np.float32)
        range_xyz = np.full((self.H, self.W, 3), 0, dtype=np.float32)
        range_sem = np.full((self.H, self.W), 0, dtype=np.uint8)

        # points += self.lidar_position
        # points[:, 1] *= -1

        range_depth[proj_h, proj_w] = depth
        range_xyz[proj_h, proj_w] = points
        range_sem[proj_h, proj_w] = semantics
        return range_depth, range_xyz, range_sem
