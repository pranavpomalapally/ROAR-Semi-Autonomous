import logging

from ROAR.agent_module.agent import Agent
from ROAR.utilities_module.data_structures_models import SensorsData, Transform, Location
from ROAR.utilities_module.vehicle_models import Vehicle, VehicleControl
from ROAR.configurations.configuration import Configuration as AgentConfig
import cv2
import numpy as np
from enum import Enum
from typing import Optional, List, Tuple
from collections import deque
from ROAR.control_module.intel_racing_pid_controller import IntelRacingImagePIDController as ImageBasedPIDController
from ROAR.control_module.intel_racing_pid_controller import \
    IntelRacingWaypointPIDController as WaypointBasedPIDController
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from datetime import datetime
import requests
import open3d as o3d
from ROAR.perception_module.depth_to_pointcloud_detector import DepthToPointCloudDetector


class PointcloudRecordingAgent(Agent):
    def __init__(self, vehicle: Vehicle, agent_settings: AgentConfig, **kwargs):
        super().__init__(vehicle, agent_settings, **kwargs)

        # occupancy grid map
        # point cloud visualization
        self.should_visualize_points = True
        if self.should_visualize_points:
            self.vis = o3d.visualization.Visualizer()
            self.vis.create_window(width=500, height=500)
            self.pcd = o3d.geometry.PointCloud()
            self.coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
            self.points_added = False

        # pointcloud and ground plane detection
        self.depth2pointcloud = DepthToPointCloudDetector(agent=self)
        self.max_dist = 1.5
        self.height_threshold = 1
        self.ransac_dist_threshold = 0.01
        self.ransac_n = 3
        self.ransac_itr = 100

        self.waypoint_map: Optional[Map] = None
        buffer = 10
        x_scale = 20
        y_scale = 20
        x_offset = 100
        y_offset = 100
        self.occu_map = Map(
            x_offset=x_offset, y_offset=y_offset, x_scale=x_scale, y_scale=y_scale,
            x_width=2500, y_height=2500, buffer=10, name="occupancy map"
        )
        self.m = np.zeros(shape=(self.occu_map.map.shape[0], self.occu_map.map.shape[1], 3))

    def run_step(self, sensors_data: SensorsData, vehicle: Vehicle) -> VehicleControl:
        super(PointcloudRecordingAgent, self).run_step(sensors_data, vehicle)
        if self.front_rgb_camera.data is not None and self.front_depth_camera.data is not None:
            try:
                pcd: o3d.geometry.PointCloud = self.depth2pointcloud.run_in_series(self.front_depth_camera.data,
                                                                                   self.front_rgb_camera.data)
                folder_name = Path("./data/pointcloud")
                folder_name.mkdir(parents=True, exist_ok=True)
                o3d.io.write_point_cloud((folder_name / f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S_%f')}.pcd").as_posix(),
                                         pcd, print_progress=True)

                pcd = self.filter_ground(pcd)

                points = np.asarray(pcd.points)
                new_points = np.copy(points)

                points = np.vstack([new_points[:, 0], new_points[:, 2]]).T

                self.occu_map.update(points, val=1)
                coord = self.occu_map.world_loc_to_occu_map_coord(loc=self.vehicle.transform.location)
                self.m[np.where(self.occu_map.map == 1)] = [255, 255, 255]
                self.m[coord[1] - 2:coord[1] + 2, coord[0] - 2:coord[0] + 2] = [0, 0, 255]
                cv2.imshow("m", self.m)
            except Exception as e:
                print(e)
        return VehicleControl()

    def filter_ground(self, pcd: o3d.geometry.PointCloud, max_dist: float = -1, height_threshold=0.1,
                      ransac_dist_threshold=0.01, ransac_n=3, ransac_itr=100) -> o3d.geometry.PointCloud:
        """
        Find ground from point cloud by first selecting points that are below the (car's position + a certain threshold)
        Then it will take only the points that are less than `max_dist` distance away
        Then do RANSAC on the resulting point cloud.

        Note that this function assumes that the ground will be the largest plane seen after filtering out everything
        above the vehicle

        Args:
            pcd: point cloud to be parsed
            max_dist: maximum distance
            height_threshold: additional height padding
            ransac_dist_threshold: RANSAC distance threshold
            ransac_n: RANSAC starting number of points
            ransac_itr: RANSAC number of iterations

        Returns:
            point cloud that only has the ground.

        """

        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)
        # height and distance filter
        # 0 -> left and right | 1 -> up and down | 2 = close and far
        points_of_interest = np.where((points[:, 1] < 0.3))
        points = points[points_of_interest]
        colors = colors[points_of_interest]
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        plane_model, inliers = pcd.segment_plane(distance_threshold=ransac_dist_threshold,
                                                 ransac_n=ransac_n,
                                                 num_iterations=ransac_itr)

        pcd: o3d.geometry.PointCloud = pcd.select_by_index(inliers)
        pcd = pcd.voxel_down_sample(0.01)
        return pcd

    def non_blocking_pcd_visualization(self, pcd: o3d.geometry.PointCloud,
                                       should_center=False,
                                       should_show_axis=False,
                                       axis_size: float = 1):
        """
        Real time point cloud visualization.

        Args:
            pcd: point cloud to be visualized
            should_center: true to always center the point cloud
            should_show_axis: true to show axis
            axis_size: adjust axis size

        Returns:
            None

        """
        if self.should_visualize_points:
            points = np.asarray(pcd.points)
            colors = np.asarray(pcd.colors)
            if should_center:
                points = points - np.mean(points, axis=0)

            if self.points_added is False:
                self.pcd = o3d.geometry.PointCloud()
                self.pcd.points = o3d.utility.Vector3dVector(points)
                self.pcd.colors = o3d.utility.Vector3dVector(colors)

                if should_show_axis:
                    self.coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_size,
                                                                                              origin=np.mean(points,
                                                                                                             axis=0))
                    self.vis.add_geometry(self.coordinate_frame)
                self.vis.add_geometry(self.pcd)
                self.points_added = True
            else:
                # print(np.shape(np.vstack((np.asarray(self.pcd.points), points))))
                self.pcd.points = o3d.utility.Vector3dVector(points)
                self.pcd.colors = o3d.utility.Vector3dVector(colors)
                if should_show_axis:
                    self.coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_size,
                                                                                              origin=np.mean(points,
                                                                                                             axis=0))
                    self.vis.update_geometry(self.coordinate_frame)
                self.vis.update_geometry(self.pcd)

            self.vis.poll_events()
            self.vis.update_renderer()


class Map:
    def __init__(self,
                 x_offset: float, y_offset: float, x_scale: float, y_scale: float,
                 x_width: int = 5000, y_height: int = 5000, buffer: int = 100,
                 name: str = "map"
                 ):
        self.name = name
        self.x_offset = x_offset
        self.y_offset = y_offset
        self.x_scale = x_scale
        self.y_scale = y_scale
        self.x_width = x_width
        self.y_height = y_height
        self.buffer = buffer
        self.map = np.zeros(shape=(self.y_height, self.x_width))

    def world_loc_to_occu_map_coord(self, loc: Location) -> Tuple[int, int]:
        """
        Takes in a coordinate in the world reference frame and transform it into the occupancy map coordinate by
        applying the equation
        `int( (WORLD + OFFSET ) * SCALE)`

        Args:
            loc:

        Returns:

        """
        x = int((loc.x + self.x_offset) * self.x_scale) + self.buffer
        y = int((loc.z + self.y_offset) * self.y_scale) + self.buffer
        return x, y

    def world_arr_to_occu_map(self, arr: np.ndarray) -> np.ndarray:
        xs = ((arr[:, 0] + self.x_offset) * self.x_scale + self.buffer).astype(int)
        ys = ((arr[:, 1] + self.y_offset) * self.y_scale + self.buffer).astype(int)
        return np.array([xs, ys]).T

    def update(self, points: np.ndarray, val=1) -> int:
        """

        Args:
            val: value to update those points to
            points: points is a 2D numpy array consist of X and Z coordinates

        Returns:
            number of points updated
        """
        # print(np.min(points, axis=0), np.max(points, axis=0))

        points = self.world_arr_to_occu_map(points)
        self.map = np.zeros(shape=self.map.shape)
        self.map[points[:, 1], points[:, 0]] = val
        return len(points)

    def visualize(self, dsize: Optional[Tuple] = None):
        img = self.map.copy()
        if dsize:
            img = cv2.resize(img, dsize=dsize)
        cv2.imshow(self.name, img)

    @staticmethod
    def filter_outlier(track,
                       min_distance_btw_points: float = 0,
                       max_distance_btw_points: float = 0.2):
        filtered = []
        max_num_points_skipped = 0
        num_points_skipped = 0
        filtered.append(track[0])
        for i in range(1, len(track)):
            x2, z2 = track[i]
            x1, z1 = filtered[-1]
            diff_x, diff_z = abs(x2 - x1), abs(z2 - z1)
            if min_distance_btw_points < diff_x < max_distance_btw_points and min_distance_btw_points < diff_z < max_distance_btw_points:
                filtered.append([x2, z2])
                num_points_skipped = 0
            else:
                num_points_skipped += 1

            max_num_points_skipped = max(num_points_skipped, max_num_points_skipped)

        filtered = np.array(filtered)
        return filtered
