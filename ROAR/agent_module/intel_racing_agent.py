from ROAR.agent_module.agent import Agent
from ROAR.utilities_module.data_structures_models import SensorsData, Transform, Location
from ROAR.utilities_module.vehicle_models import Vehicle, VehicleControl
from ROAR.configurations.configuration import Configuration as AgentConfig
import cv2
import numpy as np
from ROAR.control_module.intel_racing_pid_controller import IntelRacingPIDController as LaneFollowingPID
from collections import deque
from typing import List, Tuple, Optional

from ROAR.perception_module.depth_to_pointcloud_detector import DepthToPointCloudDetector
import open3d as o3d
import math
from enum import Enum



class IntelRacingAgent(Agent):
    def __init__(self, vehicle: Vehicle, agent_settings: AgentConfig, **kwargs):
        super().__init__(vehicle, agent_settings, **kwargs)
        self.is_lead_car = True
        self.name = "car_0"
        self.car_to_follow = "car_1"
        self.controller = LaneFollowingPID(agent=self)
        self.prev_steerings: deque = deque(maxlen=10)

        # point cloud visualization
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(width=500, height=500)
        self.pcd = o3d.geometry.PointCloud()
        self.coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
        self.points_added = False

        # pointcloud and ground plane detection
        self.depth2pointcloud = DepthToPointCloudDetector(agent=self)
        self.max_dist = 1.5
        self.height_threshold = 0.5
        self.ransac_dist_threshold = 0.01
        self.ransac_n = 3
        self.ransac_itr = 100

        # occupancy map
        self.scaling_factor = 100
        self.occu_map = np.zeros(shape=(math.ceil(self.max_dist * self.scaling_factor),
                                        math.ceil(self.max_dist * self.scaling_factor)),
                                 dtype=np.float32)
        self.cx = len(self.occu_map) // 2
        self.cz = 0

    def run_step(self, sensors_data: SensorsData, vehicle: Vehicle) -> VehicleControl:
        super().run_step(sensors_data=sensors_data, vehicle=vehicle)
        return self.lead_car_step()

    def lead_car_step(self):
        if self.front_depth_camera.data is not None and self.front_rgb_camera.data is not None:
            # if I have a camera feed
            error = self.find_error() # use lane detection to find error
            if error is None: # if i did not see a lane
                return self.no_line_seen() 
            else:
                # if i saw a lane
                self.kwargs["lat_error"] = error
                # let the agent to `remember` the error and let the PID controller to decide how much throttle and steering to give
                self.vehicle.control = self.controller.run_in_series(next_waypoint=None)
                self.prev_steerings.append(self.vehicle.control.steering)
                return self.vehicle.control
        else:
            # image feed is not available yet
            return VehicleControl()

    def no_line_seen(self):
        # did not see the line
        neutral = -90
        incline = self.vehicle.transform.rotation.pitch - neutral
        if incline < -10:
            # is down slope, execute previous command as-is
            # get the PID for downhill
            long_control = self.controller.long_pid_control()
            self.vehicle.control.throttle = long_control
            return self.vehicle.control

        else:
            # is flat or up slope, execute adjusted previous command
            return self.execute_prev_command()

    def find_error(self):
        # make rgb and depth into the same shape
        data: np.ndarray = cv2.resize(self.front_rgb_camera.data.copy(),
                                      dsize=(192, 256))
        # cv2.imshow("rgb_mask", cv2.inRange(data, self.rgb_lower_range, self.rgb_upper_range))
        data = self.rgb2ycbcr(data)
        # cv2.imshow("ycbcr_mask", cv2.inRange(data, self.ycbcr_lower_range, self.ycbcr_upper_range))
        # find the lane
        error_at_10 = self.find_error_at(data=data,
                                         y_offset=10,
                                         error_scaling=[
                                             (20, 0.1),
                                             (40, 0.75),
                                             (60, 0.8),
                                             (80, 0.9),
                                             (100, 0.95),
                                             (200, 1)
                                         ])
        error_at_50 = self.find_error_at(data=data,
                                         y_offset=50,
                                         error_scaling=[
                                             (20, 0.2),
                                             (40, 0.4),
                                             (60, 0.7),
                                             (70, 0.7),
                                             (80, 0.7),
                                             (100, 0.8),
                                             (200, 0.8)
                                         ]
                                         )

        if error_at_10 is None and error_at_50 is None:
            return None

        # we only want to follow the furthest thing we see.
        error = 0
        if error_at_10 is not None:
            error = error_at_10
        if error_at_50 is not None:
            error = error_at_50
        return error

    def find_error_at(self, data, y_offset, error_scaling) -> Optional[float]:
        y = data.shape[0] - y_offset
        lane_x = []
        # cv2.imshow("data", data)
        # mask_red = cv2.inRange(src=data, lowerb=(0, 150, 60), upperb=(250, 240, 140))  # TERRACE RED
        # mask_yellow = cv2.inRange(src=data, lowerb=(0, 130, 0), upperb=(250, 200, 110)) # TERRACE YELLOW
        # mask_red = cv2.inRange(src=data, lowerb=(0, 180, 60), upperb=(250, 240, 140))  # CORY 337 RED
        mask_yellow = cv2.inRange(src=data, lowerb=(0, 140, 0), upperb=(250, 200, 80))  # CORY 337 YELLOW
        mask = mask_yellow
        # mask = mask_red | mask_yellow

        # cv2.imshow("Lane Mask (Red)", mask_red)
        # cv2.imshow("Lane Mask (Yellow)", mask_yellow)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=1)
        cv2.imshow("Lane Mask (Yellow + Red)", mask)

        for x in range(0, data.shape[1], 5):
            if mask[y][x] > 0:
                lane_x.append(x)

        if len(lane_x) == 0:
            return None

        # if lane is found
        avg_x = int(np.average(lane_x))

        # find error
        center_x = data.shape[1] // 2

        error = avg_x - center_x
        # we want small error to be almost ignored, only big errors matter.
        for e, scale in error_scaling:
            if abs(error) <= e:
                # print(f"Error at {y_offset} -> {error, scale} -> {error * scale}")
                error = error * scale
                break

        return error

    def execute_prev_command(self):
        # no lane found, execute the previous control with a decaying factor
        if np.average(self.prev_steerings) < 0:
            self.vehicle.control.steering = -1
        else:
            self.vehicle.control.steering = 1
        # self.logger.info("Cannot see line, executing prev cmd")
        self.prev_steerings.append(self.vehicle.control.steering)
        self.vehicle.control.throttle = self.controller.long_pid_control()
        # self.logger.info(f"No Lane found, executing discounted prev command: {self.vehicle.control}")
        return self.vehicle.control

    def rgb2ycbcr(self, im):
        xform = np.array([[.299, .587, .114],
                          [-.1687, -.3313, .5],
                          [.5, -.4187, -.0813]])
        ycbcr = im.dot(xform.T)
        ycbcr[:, :, [1, 2]] += 128
        return np.uint8(ycbcr)
