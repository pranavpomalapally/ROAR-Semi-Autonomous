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
from ROAR.control_module.real_world_waypoint_based_pid_controller import RealWorldWaypointPIDController
from pathlib import Path
from datetime import datetime
import requests
import open3d as o3d


class RealLifeWaypointFollowerAgentModes(Enum):
    STOP_MID = 1
    WAYPOINT_FOLLOWING = 2
    STOP_END = 3


class RealLifeWaypointFollowerAgent(Agent):
    def __init__(self, vehicle: Vehicle, agent_settings: AgentConfig, **kwargs):
        super().__init__(vehicle, agent_settings, **kwargs)
        self.mode = RealLifeWaypointFollowerAgentModes.STOP_MID

        # Waypoint Following
        self.start_loc: Optional[Transform] = None
        self.curr_waypoint_index = 0
        self.closeness_threshold = 0.4
        self.waypoint_start_time = 0
        self.wait_time = 200
        self.waypoint_map: Optional[Map] = None
        self.waypoints: List[Transform] = self.read_waypoints(Path("./data/transforms.txt"))
        waypoints_arr = np.array([[w.location.x, w.location.z] for w in self.waypoints])
        buffer = 10
        x_scale = 20
        y_scale = 20
        x_offset = abs(min(waypoints_arr[:, 0]))
        y_offset = abs(min(waypoints_arr[:, 1]))
        width = int((max(waypoints_arr[:, 0]) - min(waypoints_arr[:, 0])) * x_scale + x_offset + buffer)
        height = int((max(waypoints_arr[:, 1]) - min(waypoints_arr[:, 1])) * y_scale + y_offset + buffer)
        self.waypoint_map = Map(x_offset=x_offset, y_offset=y_offset, x_scale=x_scale, y_scale=y_scale,
                                x_width=width, y_height=height, buffer=buffer)
        self.waypoint_map.update(waypoints_arr)

        self.controller = RealWorldWaypointPIDController(agent=self)

    def run_step(self, sensors_data: SensorsData,
                 vehicle: Vehicle) -> VehicleControl:
        super(RealLifeWaypointFollowerAgent, self).run_step(sensors_data=sensors_data, vehicle=vehicle)

        if self.mode == RealLifeWaypointFollowerAgentModes.STOP_MID:
            return self.on_STOP_MID_step()
        elif self.mode == RealLifeWaypointFollowerAgentModes.WAYPOINT_FOLLOWING:
            return self.on_WAYPOINT_FOLLOWING_step()
        elif self.mode == RealLifeWaypointFollowerAgentModes.STOP_END:
            return self.on_STOP_END_step()
        else:
            self.logger.error(f"Unknown mode detected: {self.mode}")
            return VehicleControl()

    @staticmethod
    def read_waypoints(path: Path) -> List[Transform]:
        assert path.exists(), f"Path {path} does not exist"
        file = path.open(mode='r')
        result = []
        with open(path.as_posix(), "r") as f:
            for line in f:
                result.append(RealLifeWaypointFollowerAgent._read_line(line=line))

        result = [RealLifeWaypointFollowerAgent._raw_coord_to_transform(item) for item in result]
        return result

    @staticmethod
    def _raw_coord_to_transform(raw: List[float]) -> Optional[Transform]:
        """
        transform coordinate to Transform instance
        Args:
            raw: coordinate in form of [x, y, z, pitch, yaw, roll]
        Returns:
            Transform instance
        """
        return Transform(
            location=Location(x=raw[0], y=raw[1], z=raw[2]),
        )

    @staticmethod
    def _read_line(line: str) -> List[float]:
        """
        parse a line of string of "x,y,z" into [x,y,z]
        Args:
            line: comma delimetered line
        Returns:
            [x, y, z]
        """
        try:
            x, y, z = line.split(",")
            x, y, z = float(x), float(y), float(z)
            return [x, y, z]
        except:
            x, y, z, roll, pitch, yaw = line.split(",")
            return [float(x), float(y), float(z), float(roll), float(pitch), float(yaw)]

    def on_STOP_MID_step(self):
        self.logger.info("STOP MID step")
        if self.vehicle.transform.location.x != 0:
            # go back to lane following
            self.agent_settings.pid_config_file_path = (Path(self.agent_settings.pid_config_file_path).parent /
                                                        "iOS_image_pid_config.json").as_posix()
            self.mode = RealLifeWaypointFollowerAgentModes.WAYPOINT_FOLLOWING

            # find the correct starting waypoint index by finding the closest waypoint to the vehicle
            closest_dist = 100000
            closest_index = -1
            for i in range(len(self.waypoints)):
                loc = self.waypoints[i].location
                dist = loc.distance(self.vehicle.transform.location)
                if dist < closest_dist:
                    closest_index = i
                    closest_dist = dist
            self.curr_waypoint_index = closest_index
            self.logger.info(f"Starting at index {self.curr_waypoint_index} -> {self.waypoints[closest_index]}")
            self.start_loc = self.vehicle.transform
        return VehicleControl()

    def on_WAYPOINT_FOLLOWING_step(self):
        self.logger.info("WAYPOINT_FOLLOWING step")
        control = VehicleControl()

        # find next waypoint
        next_waypoint, index = self.find_next_waypoint(waypoints_queue=self.waypoints,
                                                       curr_index=self.curr_waypoint_index,
                                                       curr_transform=self.vehicle.transform,
                                                       closeness_threshold=self.closeness_threshold)

        self.curr_waypoint_index = index
        # waypoint based pid
        control = self.controller.run_in_series(next_waypoint=next_waypoint)

        self.waypoint_visualize(map_data=self.waypoint_map.map.copy(),
                                name="waypoint visualization",
                                next_waypoint_location=next_waypoint.location,
                                car_location=self.vehicle.transform.location)

        if self.is_within_start_loc(target_location=self.start_loc.location,
                                    current_location=self.vehicle.transform.location) and \
                abs(self.time_counter - self.waypoint_start_time) > self.wait_time:
            print(self.time_counter, self.waypoint_start_time, self.wait_time)
            self.mode = RealLifeWaypointFollowerAgentModes.STOP_END
        return control

    def on_STOP_END_step(self):
        self.logger.info("STOP END step")

        return VehicleControl()

    def waypoint_visualize(self,
                           map_data: np.ndarray,
                           name: str = "waypoint_visualization",
                           car_location: Optional[Location] = None,
                           next_waypoint_location: Optional[Location] = None):
        m = np.zeros(shape=(map_data.shape[0], map_data.shape[1], 3))
        m[np.where(map_data > 0.9)] = [255, 255, 255]
        point_size = 2
        if car_location is not None:
            coord = self.waypoint_map.world_loc_to_occu_map_coord(car_location)
            m[coord[1] - point_size:coord[1] + point_size, coord[0] - point_size:coord[0] + point_size] = [0, 0, 255]

        if next_waypoint_location is not None:
            coord = self.waypoint_map.world_loc_to_occu_map_coord(next_waypoint_location)
            m[coord[1] - point_size:coord[1] + point_size, coord[0] - point_size:coord[0] + point_size] = [0, 255, 0]
        cv2.imshow(name, m)
        cv2.waitKey(1)

    @staticmethod
    def find_next_waypoint(waypoints_queue: List[Transform], curr_index, curr_transform: Transform,
                           closeness_threshold) -> \
            Tuple[Transform, int]:
        assert curr_index < len(waypoints_queue), f"Trying to compute next waypoint starting at {curr_index} " \
                                                  f"in an array of size {len(waypoints_queue)}"

        while True:
            waypoint: Transform = waypoints_queue[curr_index]
            curr_dist = curr_transform.location.distance(waypoint.location)
            if curr_dist > closeness_threshold:
                break
            else:
                curr_index = (curr_index + 1) % len(waypoints_queue)
        return waypoints_queue[curr_index], curr_index

    @staticmethod
    def is_within_start_loc(target_location: Location, current_location: Location, bound: float = 0.5):
        if target_location.x - bound < current_location.x < target_location.x + bound:
            if target_location.z - bound < current_location.z < target_location.z + bound:
                return True
        return False


class Map:
    def __init__(self,
                 x_offset: float, y_offset: float, x_scale: float, y_scale: float,
                 x_width: int = 5000, y_height: int = 5000, buffer: int = 100,
                 name: str = "map"
                 ):
        self.name = name
        # Waypoint Following
        self.waypoints: List[Transform] = []
        self.curr_waypoint_index = 0
        self.closeness_threshold = 0.4
        self.waypoint_start_time = 0

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
