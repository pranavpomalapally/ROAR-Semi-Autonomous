"""
This agent will demonstrate automatic free space driving using
1. Detecting ground plane
2. Find free space
3. Find next waypoint on the free space
4. drive as smooth as possible toward that waypoint

"""
import time

from ROAR.agent_module.agent import Agent
from ROAR.utilities_module.data_structures_models import SensorsData
from ROAR.utilities_module.vehicle_models import Vehicle, VehicleControl
from ROAR.configurations.configuration import Configuration as AgentConfig
import cv2
import numpy as np
import open3d as o3d
from ROAR.utilities_module.occupancy_map import OccupancyGridMap
from ROAR.perception_module.depth_to_pointcloud_detector import DepthToPointCloudDetector
from ROAR.perception_module.ground_plane_detector import GroundPlaneDetector
from collections import deque


class RLPrepAgent(Agent):
    def __init__(self, vehicle: Vehicle, agent_settings: AgentConfig, **kwargs):
        super().__init__(vehicle, agent_settings, **kwargs)

        # initialize occupancy grid map content
        self.occu_map = OccupancyGridMap(agent=self)
        self.depth_to_pcd = DepthToPointCloudDetector(agent=self)
        self.ground_plane_detector = GroundPlaneDetector(agent=self)

        self.queue = deque(maxlen=4)

    def run_step(self, sensors_data: SensorsData, vehicle: Vehicle) -> VehicleControl:
        super().run_step(sensors_data, vehicle)
        if self.front_depth_camera.data is not None and self.front_rgb_camera.data is not None:
            points = self.depth_to_pcd.run_in_series()
            # find plane
            output = self.ground_plane_detector.run_in_series(points=points)
            if output is not None:
                inliers = output
                points: np.ndarray = points[inliers]
                points = points[np.random.choice(points.shape[0], 2000, replace=False)]
                self.occu_map.update(points)
                self.occu_map.visualize()
                if self.time_counter % 5 == 0:
                    self.add_one_dataset()

            if len(self.queue) == 4:
                imgs = []
                for i in range(len(self.queue)):
                    imgs.append(np.concatenate((self.queue[i][0], self.queue[i][1], self.queue[i][2]), axis=0))
                img = np.concatenate([im for im in imgs], axis=1)

                cv2.imshow("Data Aggregated", img)

        return VehicleControl()

    def add_one_dataset(self, view_size=(100, 100)):
        # TODO center graph on the bottom instead of around the vehicle
        # occupancy map
        view_offset = 10
        occu_map_cropped = self.occu_map.get_map(transform=self.vehicle.transform, view_size=view_size)
        occu_map_cropped = occu_map_cropped[0:occu_map_cropped.shape[1]//2+view_offset, :]
        occu_map_cropped = self.clean_occu_map(occu_map_cropped, kernel_size=(3, 3))
        # vehicle location map
        vehicle_loc_map = np.zeros(occu_map_cropped.shape)
        x, y = vehicle_loc_map.shape[0] - 10, vehicle_loc_map.shape[1]//2
        vehicle_loc_map[np.where(occu_map_cropped > 0)] = 0.5
        vehicle_loc_map[x - 2:x + 2, y - 2:y + 2] = 1

        # reward map (higher as you go further forward)
        all_road_reward = np.where(occu_map_cropped > 0.5)
        y_coords_in_front_of_car = all_road_reward[0] < 50
        all_road_reward = all_road_reward[0][y_coords_in_front_of_car], all_road_reward[1][y_coords_in_front_of_car]
        distances = np.sqrt(np.square(all_road_reward[0] - x) + np.square(all_road_reward[1] - y))
        reward_map = np.zeros(shape=occu_map_cropped.shape)
        max_dist = np.sqrt((vehicle_loc_map.shape[0] - x) ** 2 + (vehicle_loc_map.shape[1] - y) ** 2)
        reward_map[all_road_reward] = distances / max_dist
        # add reward map, vehicle location map, and occu map to queue
        self.queue.appendleft((occu_map_cropped, vehicle_loc_map, reward_map))

    @staticmethod
    def clean_occu_map(data, kernel_size=(2, 2)):
        kernel = np.ones(kernel_size, np.uint8)
        erosion = cv2.erode(data, kernel, iterations=3)
        dilation = cv2.dilate(erosion, kernel, iterations=1)
        return dilation

    def non_blocking_pcd_visualization(self, pcd: o3d.geometry.PointCloud,
                                       should_center=False,
                                       should_show_axis=False,
                                       axis_size: float = 0.1):
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
