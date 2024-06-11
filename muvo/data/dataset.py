import os
from glob import glob
from PIL import Image
import csv
import traceback

import numpy as np
import pandas as pd
import lightning.pytorch as pl
import scipy.ndimage
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

from constants import CARLA_FPS, EGO_VEHICLE_DIMENSION, LABEL_MAP, VOXEL_LABEL, VOXEL_LABEL_CARLA
from muvo.data.dataset_utils import integer_to_binary, calculate_birdview_labels, calculate_instance_mask
from muvo.utils.geometry_utils import get_out_of_view_mask, calculate_geometry, lidar_to_histogram_features
from muvo.utils.geometry_utils import PointCloud
from data.data_preprocessing import convert_coor_lidar

class DataModule(pl.LightningDataModule):
    last_sem_depth_array = None
    last_semantic_array = None
    
    # The following attributes may be changed by prediction.py!
    initial_index = 100
    w_abs = 0.25
    w_mse = 0.25
    w_ssim = 0.25
    w_per = 0.25
    w_temp = 0.0
    threshold = 0.3346 

    def __init__(self, cfg, dataset_root=None):
        super().__init__()
        self.cfg = cfg
        self.batch_size = self.cfg.BATCHSIZE
        self.sequence_length = self.cfg.RECEPTIVE_FIELD + self.cfg.FUTURE_HORIZON

        self.dataset_root = dataset_root if dataset_root else self.cfg.DATASET.DATAROOT

        # Will be populated with self.setup()
        self.train_dataset, self.val_dataset_0, self.val_dataset_1, self.val_dataset_2 = None, None, None, None
        self.test_dataset = None

    def setup(self, stage=None):
        self.train_dataset = AnovoxDataset(self.cfg, '/path_to_AnoVox/Final_Output_2024_02_22-13_38/')
        self.val_dataset_0 = self.train_dataset
        self.val_dataset_1 = self.train_dataset
        self.val_dataset_2 = self.train_dataset
        self.test_dataset = self.train_dataset

        print(f'{len(self.train_dataset)} data points in {self.train_dataset.dataset_path}')
        # print(f'{len(self.val_dataset)} data points in {self.val_dataset.dataset_path}')
        print(f'{len(self.val_dataset_0)} data points in {self.val_dataset_0.dataset_path}')
        print(f'{len(self.val_dataset_1)} data points in {self.val_dataset_1.dataset_path}')
        print(f'{len(self.val_dataset_2)} data points in {self.val_dataset_2.dataset_path}')
        print(f'{len(self.test_dataset)} data points in prediction')

        self.train_sampler = None
        self.val_sampler_0 = range(0, len(self.val_dataset_0), 50)
        self.val_sampler_1 = range(1500, len(self.val_dataset_1), 50)
        self.val_sampler_2 = range(3000, len(self.val_dataset_2), 50)
        self.test_sampler_0 = range(0, len(self.test_dataset), 900)
        self.test_sampler_1 = range(1500, len(self.test_dataset), 600)
        self.test_sampler_2 = range(0, len(self.test_dataset), 10)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.train_sampler is None,
            num_workers=self.cfg.N_WORKERS,
            pin_memory=True,
            drop_last=True,
            sampler=self.train_sampler,
        )

    def val_dataloader(self):
        return [
            DataLoader(
                self.val_dataset_0,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.cfg.N_WORKERS,
                pin_memory=True,
                drop_last=True,
                sampler=self.val_sampler_0,
            ),
            DataLoader(
                self.val_dataset_1,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.cfg.N_WORKERS,
                pin_memory=True,
                drop_last=True,
                sampler=self.val_sampler_1,
            ),
            DataLoader(
                self.val_dataset_2,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.cfg.N_WORKERS,
                pin_memory=True,
                drop_last=True,
                sampler=self.val_sampler_2,
            ),
        ]

    def test_dataloader(self):
        return [
            DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.cfg.N_WORKERS,
                pin_memory=True,
                drop_last=True,
                sampler=self.test_sampler_0,
        ),
            DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.cfg.N_WORKERS,
                pin_memory=True,
                drop_last=True,
                sampler=self.test_sampler_1,
            ),
            DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.cfg.N_WORKERS,
                pin_memory=True,
                drop_last=True,
                sampler=self.test_sampler_2,
            ),
        ]


class AnovoxDataset(Dataset):
    def __init__(self, cfg, dataset_root=None, runs_filter='*', img_base='RGB_IMG', folder_exclude='Scenario_Configuration_Files', file_exclude='color_palette.txt', scenario_filter='*'):
        self.cfg = cfg
        self.sequence_length = self.cfg.RECEPTIVE_FIELD + self.cfg.FUTURE_HORIZON

        self.dataset_path = os.path.join(dataset_root)
        self.intrinsics, self.extrinsics = calculate_geometry_from_config(self.cfg)

        # Iterate over all runs in the data folder
        runs = sorted(glob(os.path.join(self.dataset_path, scenario_filter)))

        #Filter out non-scenario folder and file in AnoVox dataset
        runs = [run for run in runs if folder_exclude not in run and file_exclude not in run]

        self.image_paths = []
        self.csv_paths = []
        self.routemap_paths = []
        self.birdview_paths = []
        self.point_clouds_paths = []
        self.sem_point_cloud_paths = []
        self.instance_segmentation_paths = []
        self.semantic_segmentation_paths = []
        for run_path in runs:
            run = os.path.basename(run_path)

            #Load all the data
            images = sorted(glob(os.path.join(self.dataset_path, run, img_base, '*.png')))
            csv_files = sorted(glob(os.path.join(self.dataset_path, run, 'ACTION', '*.csv')))
            route_maps = sorted(glob(os.path.join(self.dataset_path, run, 'ROUTE_MAP', '*.png')))
            birdviews = sorted(glob(os.path.join(self.dataset_path, run, 'birdview', '*.png')))
            point_clouds = sorted(glob(os.path.join(self.dataset_path, run, 'PCD', '*.npy')))
            semantic_point_clouds = sorted(glob(os.path.join(self.dataset_path, run, 'SEMANTIC_PCD', '*.npy')))
            instances = sorted(glob(os.path.join(self.dataset_path, run, 'INSTANCE_IMG', '*.png')))
            semantic_maps = sorted(glob(os.path.join(self.dataset_path, run, 'SEMANTIC_IMG', '*.png')))
            self.csv_paths.append(csv_files)
            self.image_paths.append(images)
            self.routemap_paths.append(route_maps)
            self.birdview_paths.append(birdviews)
            self.point_clouds_paths.append(point_clouds)
            self.sem_point_cloud_paths.append(semantic_point_clouds)
            self.instance_segmentation_paths.append(instances)
            self.semantic_segmentation_paths.append(semantic_maps)

        self.data_pointers = self.get_data_pointers()

    def get_data_pointers(self):
        data_pointers = []

        for i, scenario in enumerate(self.image_paths):
            # Calculate normalised reward of the run
            run_length = len(scenario)
            scenario_index = i

            stride = int(self.cfg.DATASET.STRIDE_SEC * CARLA_FPS)
            # Loop across all elements in the dataset, and make all elements in a sequence belong to the same run
            start_index = DataModule.initial_index
            total_length = run_length - stride * self.sequence_length
            print(total_length)
            for i in range(start_index, total_length):
                frame_indices = range(i, i + stride * self.sequence_length, stride)
                data_pointers.append((scenario_index, list(frame_indices)))

        return data_pointers
    
    def __getitem__(self, i):
        batch = {}

        run_id, indices = self.data_pointers[i]
        for t in indices:
            try:
                single_element_t = self.load_single_element_time_t(run_id, t)
            except Exception as e:
                print(e)
                print(traceback.format_exc())
                print(f'{run_id}, {t} data is invalid')
                continue

            for k, v in single_element_t.items():
                batch[k] = batch.get(k, []) + [v]

        for k, v in batch.items():
            batch[k] = torch.from_numpy(np.stack(v))

        return batch
    
    def load_single_element_time_t(self, run_id, t):
        data_path = self.image_paths[run_id][t]
        single_element_t = {}

        # Load image
        image = Image.open(data_path)
        image = np.asarray(image).transpose((2, 0, 1))
        single_element_t['image'] = image

        instance_segmentation_map = Image.open(self.instance_segmentation_paths[run_id][t])
        instance_segmentation_map = np.asfarray(instance_segmentation_map)
        single_element_t['instance_segmentation_map'] = instance_segmentation_map

        sem_seg_map = Image.open(self.semantic_segmentation_paths[run_id][t])
        sem_seg_map.convert('RGB')
        sem_seg_map = np.asarray(sem_seg_map)
        single_element_t['semantic_segmentation_map'] = sem_seg_map

        route_map = Image.open(self.routemap_paths[run_id][t])
        route_map = np.asarray(route_map)[None]
        # Make the grayscale image an RGB image
        _, h, w = route_map.shape
        route_map = np.broadcast_to(route_map, (3, h, w)).copy()
        single_element_t['route_map'] = route_map

        birdview = np.asarray(Image.open(self.birdview_paths[run_id][t]))
        h, w = birdview.shape
        n_classes = 10
        birdview = integer_to_binary(birdview.reshape(-1), n_classes).reshape(h, w, n_classes)
        birdview = birdview.transpose((2, 0, 1))
        single_element_t['birdview'] = birdview
        birdview_label = calculate_birdview_labels(torch.from_numpy(birdview), n_classes).numpy()
        birdview_label = birdview_label[None]
        single_element_t['birdview_label'] = birdview_label
        
        points = np.load(self.point_clouds_paths[run_id][t])[:,:3]

        # mask ego-vehicle
        x, y, z = EGO_VEHICLE_DIMENSION
        ego_box = np.array([[-x / 2, -y / 2, 0], [x / 2, y / 2, z]])
        ego_idx = ((ego_box[0] < points) & (points < ego_box[1])).all(axis=1)
        points = points[~ego_idx]

        # range-view of lidar point cloud
        self.pcd = PointCloud()
        range_view_pcd_depth, range_view_pcd_xyz = self.pcd.do_range_projection_w_o_sem(points)
        if self.cfg.MODEL.LIDAR.ENABLED:
            single_element_t['range_view_pcd_xyzd'] = np.concatenate(
                [range_view_pcd_xyz, range_view_pcd_depth[..., None]], axis=-1).transpose((2, 0, 1))  # x y z d

        instance_map = Image.open(self.instance_segmentation_paths[run_id][t])
        DataModule.last_sem_depth_array = np.asarray(instance_map)
        # Load action
        throttle = steering = brake = throttle_brake = speed = None
        with open(self.csv_paths[run_id][t]) as csv_file:
            reader = csv.reader(csv_file, delimiter=';')
            for line in reader:
                if(line[0] == 'throttle '):
                    throttle = float(line[1])
                elif(line[0] == 'brake '):
                    brake = float(line[1])
                elif(line[0] == 'steer '):
                    steering = float(line[1])
                elif(line[0] == 'speed '):
                    speed = float(line[1])
        # throttle, steering, brake = data_row['action']
        throttle_brake = throttle if throttle > 0 else -brake

        single_element_t['steering'] = np.array([steering], dtype=np.float32)
        single_element_t['throttle_brake'] = np.array([throttle_brake], dtype=np.float32)
        single_element_t['speed'] = np.array([speed], dtype=np.float32)

        # Geometry
        single_element_t['intrinsics'] = self.intrinsics.copy()
        single_element_t['extrinsics'] = self.extrinsics.copy()

        return single_element_t
    
    def __len__(self):
        return len(self.data_pointers)



class CarlaDataset(Dataset):
    def __init__(self, cfg, mode='train', sequence_length=1, dataset_root=None, towns_filter='*', runs_filter='*'):
        self.cfg = cfg
        self.mode = mode
        self.sequence_length = sequence_length

        self.dataset_path = os.path.join(dataset_root, self.cfg.DATASET.VERSION, mode)
        self.intrinsics, self.extrinsics = calculate_geometry_from_config(self.cfg)
        self.bev_out_of_view_mask = get_out_of_view_mask(self.cfg)
        self.pcd = PointCloud(
            self.cfg.POINTS.CHANNELS,
            self.cfg.POINTS.HORIZON_RESOLUTION,
            *self.cfg.POINTS.FOV,
            self.cfg.POINTS.LIDAR_POSITION,
        )

        # Iterate over all runs in the data folder

        self.data = dict()

        towns = sorted(glob(os.path.join(self.dataset_path, towns_filter)))
        for town_path in towns:
            town = os.path.basename(town_path)

            runs = sorted(glob(os.path.join(self.dataset_path, town, runs_filter)))
            for run_path in runs:
                run = os.path.basename(run_path)
                pd_dataframe_path = os.path.join(run_path, 'pd_dataframe.pkl')

                if os.path.isfile(pd_dataframe_path):
                    self.data[f'{town}/{run}'] = pd.read_pickle(pd_dataframe_path)

        self.data_pointers = self.get_data_pointers()

    def get_data_pointers(self):
        data_pointers = []

        n_filtered_run = 0
        for run, data_run in self.data.items():
            #  Calculate normalised reward of the run
            run_length = len(data_run['reward'])
            cumulative_reward = data_run['reward'].sum()
            normalised_reward = cumulative_reward / run_length
            if normalised_reward < self.cfg.DATASET.FILTER_NORM_REWARD:
                n_filtered_run += 1
                continue

            stride = int(self.cfg.DATASET.STRIDE_SEC * CARLA_FPS)
            # Loop across all elements in the dataset, and make all elements in a sequence belong to the same run
            start_index = int(CARLA_FPS * self.cfg.DATASET.FILTER_BEGINNING_OF_RUN_SEC)
            total_length = len(data_run) - stride * self.sequence_length
            for i in range(start_index, total_length):
                frame_indices = range(i, i + stride * self.sequence_length, stride)
                data_pointers.append((run, list(frame_indices)))

        print(f'Filtered {n_filtered_run} runs in {self.dataset_path}')

        if self.cfg.EVAL.DATASET_REDUCTION:
            import random
            random.seed(0)
            final_size = int(len(data_pointers) / self.cfg.EVAL.DATASET_REDUCTION_FACTOR)
            data_pointers = random.sample(data_pointers, final_size)

        return data_pointers

    def __len__(self):
        return len(self.data_pointers)

    def __getitem__(self, i):
        batch = {}

        run_id, indices = self.data_pointers[i]
        for t in indices:
            try:
                single_element_t = self.load_single_element_time_t(run_id, t)
            except:
                print(f'{run_id}, {t} data is invalid')
                continue

            for k, v in single_element_t.items():
                batch[k] = batch.get(k, []) + [v]

        for k, v in batch.items():
            batch[k] = torch.from_numpy(np.stack(v))

        return batch

    def load_single_element_time_t(self, run_id, t):
        data_row = self.data[run_id].iloc[t]
        single_element_t = {}

        # Load image
        image = Image.open(
            os.path.join(self.dataset_path, run_id, data_row['image_path'])
        )
        image = np.asarray(image).transpose((2, 0, 1))
        single_element_t['image'] = image

        # Load route map
        route_map = Image.open(
            os.path.join(self.dataset_path, run_id, data_row['routemap_path'])
        )
        route_map = np.asarray(route_map)[None]
        # Make the grayscale image an RGB image
        _, h, w = route_map.shape
        route_map = np.broadcast_to(route_map, (3, h, w)).copy()
        single_element_t['route_map'] = route_map

        # Load bird's-eye view segmentation label
        birdview = np.asarray(Image.open(
            os.path.join(self.dataset_path, run_id, data_row['birdview_path'])
        ))
        h, w = birdview.shape
        n_classes = data_row['n_classes']
        birdview = integer_to_binary(birdview.reshape(-1), n_classes).reshape(h, w, n_classes)
        birdview = birdview.transpose((2, 0, 1))
        single_element_t['birdview'] = birdview
        birdview_label = calculate_birdview_labels(torch.from_numpy(birdview), n_classes).numpy()
        birdview_label = birdview_label[None]
        single_element_t['birdview_label'] = birdview_label

        # TODO: get person and car instance ids with json
        instance_mask = birdview[3].astype(np.bool) | birdview[4].astype(np.bool)
        instance_label, _ = scipy.ndimage.label(instance_mask[None].astype(np.int64))
        single_element_t['instance_label'] = instance_label

        # # Load lidar points clouds
        # pcd = np.load(
        #     os.path.join(self.dataset_path, run_id, data_row['points_path']),
        #     allow_pickle=True).item()  # n x 4, (x, y, z, intensity)
        # single_element_t['points_intensity'] = np.concatenate([pcd['points_xyz'], pcd['intensity'][:, None]], axis=-1)
        pcd_semantic = np.load(
            os.path.join(self.dataset_path, run_id, data_row['points_semantic_path']),
            allow_pickle=True).item()
        points = convert_coor_lidar(pcd_semantic['points_xyz'], self.cfg.POINTS.LIDAR_POSITION)

        # remap labels
        remap = np.full((max(LABEL_MAP.keys()) + 1), max(LABEL_MAP.values()), dtype=np.uint8)
        remap[list(LABEL_MAP.keys())] = list(LABEL_MAP.values())
        semantics = remap[pcd_semantic['ObjTag']]

        # mask ego-vehicle
        x, y, z = EGO_VEHICLE_DIMENSION
        ego_box = np.array([[-x / 2, -y / 2, 0], [x / 2, y / 2, z]])
        ego_idx = ((ego_box[0] < points) & (points < ego_box[1])).all(axis=1)
        semantics = semantics[~ego_idx]
        points = points[~ego_idx]

        # histogram of lidar
        # single_element_t['points'] = points
        # single_element_t['points_label'] = pcd_semantic['ObjTag'].astype('uint8')
        # single_element_t['points_histogram_xy'], \
        # single_element_t['points_histogram_xz'], \
        # single_element_t['points_histogram_yz'] = lidar_to_histogram_features(points, self.cfg)

        # range-view of lidar point cloud
        range_view_pcd_depth, range_view_pcd_xyz, range_view_pcd_sem = self.pcd.do_range_projection(points, semantics)
        if self.cfg.MODEL.LIDAR.ENABLED:
            single_element_t['range_view_pcd_xyzd'] = np.concatenate(
                [range_view_pcd_xyz, range_view_pcd_depth[..., None]], axis=-1).transpose((2, 0, 1))  # x y z d
        if self.cfg.LIDAR_SEG.ENABLED:
            single_element_t['range_view_pcd_seg'] = range_view_pcd_sem[None].astype(int)

        # data type for point-pillar
        if self.cfg.MODEL.LIDAR.POINT_PILLAR.ENABLED:
            max_num_points = int(self.cfg.POINTS.N_PER_SECOND / CARLA_FPS)
            fixed_points = np.empty((max_num_points, 3), dtype=np.float32)
            num_points = min(points.shape[0], max_num_points)
            fixed_points[:num_points] = points[:num_points]
            single_element_t['points_raw'] = fixed_points
            single_element_t['num_points'] = num_points

        # Load voxels, saved as voxel coordinates.
        if self.cfg.VOXEL_SEG.ENABLED:
            voxel_data = np.load(
                os.path.join(self.dataset_path, run_id, data_row['voxel_path'])
            )
            voxel_points = voxel_data[:, :-1]
            voxel_semantics = voxel_data[:, -1]
            voxel_semantics[voxel_semantics == 255] = 0
            voxel_semantics = remap[voxel_semantics]
            voxels = np.zeros(self.cfg.VOXEL.SIZE, dtype=np.uint8)
            voxels[voxel_points[:, 0], voxel_points[:, 1], voxel_points[:, 2]] = voxel_semantics
            single_element_t['voxel'] = voxels[None]

        # load depth and semantic image
        depth_semantic = Image.open(
            os.path.join(self.dataset_path, run_id, data_row['depth_semantic_path'])
        )
        depth_semantic = np.asarray(depth_semantic)
        DataModule.last_sem_depth_array = depth_semantic
        semantic_image = depth_semantic[..., -1]
        if self.cfg.LOSSES.RGB_INSTANCE:
            single_element_t['image_instance_mask'] = calculate_instance_mask(
                semantic_image[None],
                vehicle_idx=list(VOXEL_LABEL_CARLA.keys())[list(VOXEL_LABEL_CARLA.values()).index('Vehicle')],
                pedestrian_idx=list(VOXEL_LABEL_CARLA.keys())[list(VOXEL_LABEL_CARLA.values()).index('Pedestrian')],
            )

        # load semantic image
        if self.cfg.SEMANTIC_IMAGE.ENABLED:
            single_element_t['semantic_image'] = remap[semantic_image][None].astype(int)
        # load depth
        if self.cfg.DEPTH.ENABLED:
            depth_color = depth_semantic[..., :-1].transpose((2, 0, 1)).astype(float)
            single_element_t['depth_color'] = depth_color / 255.0
            # in carla, depth is saved in rgb-channel
            depth = (256 ** 2 * depth_color[0] + 256 * depth_color[1] + depth_color[2]) / (256 ** 3 - 1)
            depth[depth > 0.999] = -1
            single_element_t['depth'] = depth[None]

        # Load action and reward
        throttle, steering, brake = data_row['action']
        throttle_brake = throttle if throttle > 0 else -brake

        single_element_t['steering'] = np.array([steering], dtype=np.float32)
        single_element_t['throttle_brake'] = np.array([throttle_brake], dtype=np.float32)
        single_element_t['speed'] = data_row['speed']

        single_element_t['reward'] = np.array([data_row['reward']], dtype=np.float32).clip(-1.0, 1.0)
        single_element_t['value_function'] = np.array([data_row['value']], dtype=np.float32)

        # Geometry
        single_element_t['intrinsics'] = self.intrinsics.copy()
        single_element_t['extrinsics'] = self.extrinsics.copy()

        return single_element_t


def calculate_geometry_from_config(cfg):
    """ Intrinsics and extrinsics for a single camera.
    See https://github.com/bradyz/carla_utils_fork/blob/dynamic-scene/carla_utils/leaderboard/camera.py
    and https://github.com/bradyz/carla_utils_fork/blob/dynamic-scene/carla_utils/recording/sensors/camera.py
    """
    # Intrinsics
    fov = cfg.IMAGE.FOV
    h, w = cfg.IMAGE.SIZE

    # Extrinsics
    forward, right, up = cfg.IMAGE.CAMERA_POSITION
    pitch, yaw, roll = cfg.IMAGE.CAMERA_ROTATION

    return calculate_geometry(fov, h, w, forward, right, up, pitch, yaw, roll)
