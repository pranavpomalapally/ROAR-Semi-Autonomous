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


class AutoLWAgentModes(Enum):
    STOP_INIT = 1
    LANE_FOLLOWING = 2
    STOP_MID = 3
    VISUALIZE_TRACK = 4
    WAYPOINT_FOLLOWING = 5
    STOP_END = 6


class AutoLaneFollowingWithWaypointAgent(Agent):
    def __init__(self, vehicle: Vehicle, agent_settings: AgentConfig, **kwargs):
        super().__init__(vehicle, agent_settings, **kwargs)
        self.mode = AutoLWAgentModes.STOP_INIT
        self.prev_steerings: deque = deque(maxlen=10)
        self.agent_settings.pid_config_file_path = (Path(self.agent_settings.pid_config_file_path).parent /
                                                    "iOS_image_pid_config.json").as_posix()
        self.controller = ImageBasedPIDController(agent=self)

        # START LOC
        self.start_loc: Optional[Transform] = None
        self.start_loc_bound: float = 0.2
        self.has_exited_start_loc: bool = False

        # STOP Mid step
        self.ip_addr = "10.0.0.2"

        # Waypoint Following
        self.waypoints: List[Transform] = []
        self.curr_waypoint_index = 0
        self.closeness_threshold = 0.2

        # debug waypoint following
        f = Path("transforms_3.txt").open('r')
        for line in f.readlines():
            x, y, z = line.split(",")
            l = Location(x=x, y=y, z=z)
            self.waypoints.append(Transform(location=l))
        self.mode = AutoLWAgentModes.WAYPOINT_FOLLOWING

    def run_step(self, sensors_data: SensorsData, vehicle: Vehicle) -> VehicleControl:
        super(AutoLaneFollowingWithWaypointAgent, self).run_step(sensors_data, vehicle)
        if self.front_rgb_camera.data is not None and self.front_depth_camera.data is not None:
            self.prev_steerings.append(self.vehicle.control.steering)

            if self.mode == AutoLWAgentModes.STOP_INIT:
                return self.on_STOP_INIT_step()
            elif self.mode == AutoLWAgentModes.LANE_FOLLOWING:
                return self.on_LANE_FOLLOWING_step(debug=False)
            elif self.mode == AutoLWAgentModes.STOP_MID:
                return self.on_STOP_MID_step()
            elif self.mode == AutoLWAgentModes.VISUALIZE_TRACK:
                return self.on_VISUALIZE_TRACK_step()
            elif self.mode == AutoLWAgentModes.WAYPOINT_FOLLOWING:
                return self.on_WAYPOINT_FOLLOWING_step()
            elif self.mode == AutoLWAgentModes.STOP_END:
                return self.on_STOP_END_step()
            else:
                self.logger.error(f"Unknown mode detected: {self.mode}")
                return VehicleControl()
        return VehicleControl()

    def on_STOP_INIT_step(self):
        self.logger.info("STOP INIT step")
        error = self.find_error()
        if error is not None:
            self.mode = AutoLWAgentModes.LANE_FOLLOWING
            self.start_loc = self.vehicle.transform
        return VehicleControl()

    def on_LANE_FOLLOWING_step(self, debug=False):
        self.logger.info("LANE_FOLLOWING step")
        error = self.find_error()
        control = VehicleControl()
        if error is not None:
            self.kwargs["lat_error"] = error
            control = self.controller.run_in_series(next_waypoint=None)
        else:
            control = self.no_line_seen()
        if self.start_loc is not None:
            if self.is_within_start_loc(self.start_loc.location, self.vehicle.transform.location,
                                        bound=self.start_loc_bound):
                if self.has_exited_start_loc:
                    self.mode = AutoLWAgentModes.STOP_MID
                    control = VehicleControl(brake=True, throttle=0, steering=0)
                else:
                    self.logger.info("LANE_FOLLOWING step -> waiting to move outside of the initial start location: "
                                     f"{round(self.start_loc.location.x - self.start_loc_bound, 2)} < "
                                     f"{round(self.vehicle.transform.location.x, 2)} < "
                                     f"{round(self.start_loc.location.x + self.start_loc_bound, 2)} | "
                                     f"{round(self.start_loc.location.z - self.start_loc_bound, 2)} < "
                                     f"{round(self.vehicle.transform.location.z, 2)} < "
                                     f"{round(self.start_loc.location.z + self.start_loc_bound, 2)}")
            else:
                self.logger.info("LANE_FOLLOWING step -> moved outside of the initial location. "
                                 "Executing Lane Following")
                self.has_exited_start_loc = True
        if debug and self.time_counter > 200:
            self.mode = AutoLWAgentModes.STOP_MID
            control = VehicleControl(brake=True, throttle=0, steering=0)
        return control

    def on_STOP_MID_step(self):
        self.logger.info("STOP MID step")
        self.logger.info("Request sent")
        r = requests.get(f"http://{self.ip_addr}:40001/save_world")
        if r.status_code == 200:
            self.on_VISUALIZE_TRACK_step()
        else:
            # go back to lane following
            self.agent_settings.pid_config_file_path = (Path(self.agent_settings.pid_config_file_path).parent /
                                                        "iOS_image_pid_config.json").as_posix()
            self.controller = ImageBasedPIDController(agent=self)
            self.mode = AutoLWAgentModes.LANE_FOLLOWING
        return VehicleControl()

    def on_VISUALIZE_TRACK_step(self):
        self.logger.info("VISUALIZE_TRACK step")
        # take only coordinates that are valid (not initial values) by taking advantage of vehicle height
        self.transform_history = [t for t in self.transform_history if abs(t.location.y) > 0.00001]

        data = [t.location.to_array() for t in self.transform_history]
        transform_file = Path("transforms.txt").open('w+')
        for d in self.transform_history:
            transform_file.write(d.location.to_string() + "\n")
        self.logger.info("File written to transforms.txt")
        self.visualize_track_data(track_data=data)
        self.waypoints = [t for t in self.transform_history]
        self.agent_settings.pid_config_file_path = (Path(self.agent_settings.pid_config_file_path).parent /
                                                    "iOS_waypoint_pid_config.json").as_posix()
        self.controller = WaypointBasedPIDController(agent=self)
        self.mode = AutoLWAgentModes.WAYPOINT_FOLLOWING
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
        return control

    def on_STOP_END_step(self):
        self.logger.info("STOP END step")

        return VehicleControl()

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
    def visualize_track_data(track_data: List[List[float]]):
        print(f"Visualizing [{len(track_data)}] data points")

        track_data = np.asarray(track_data)
        # times = [i for i in range(len(track_data))]
        Xs = track_data[:, 0]
        Ys = track_data[:, 1]
        Zs = track_data[:, 2]

        fig = make_subplots(rows=3, cols=2, subplot_titles=["Xs", "Y vs Z", "Ys", "X vs Z", "Zs", "X vs Z"])
        fig.update_layout(
            title=f"Data file path: {datetime.now().strftime('%m_%d_%Y_%H_%M_%S_%f')}"
        )
        fig.add_trace(
            go.Scatter(
                y=Xs, mode='markers', marker=dict(color="Red"), name="Xs",
            ),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(
                y=Ys, mode='markers', marker=dict(color="Blue"), name="Ys"
            ),
            row=2, col=1
        )

        fig.add_trace(
            go.Scatter(
                y=Zs, mode='markers', marker=dict(color="Green"), name="Zs"
            ), row=3, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=Xs, y=Ys, mode='markers', marker=dict(color="Black"), name="X vs Y"
            ), row=1, col=2
        )
        fig.add_trace(
            go.Scatter(
                x=Xs, y=Zs, mode='markers', marker=dict(color="Orange"), name="X vs Z"
            ), row=2, col=2
        )
        fig.add_trace(
            go.Scatter(
                x=Ys, y=Zs, mode='markers', marker=dict(color="Yellow"), name="Y vs Z"
            ), row=3, col=2
        )

        fig.show()

    @staticmethod
    def is_within_start_loc(target_location: Location, current_location: Location, bound: float = 0.5):
        if target_location.x - bound < current_location.x < target_location.x + bound:
            if target_location.z - bound < current_location.z < target_location.z + bound:
                return True
        return False

    """
    Lane Following
    """

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
        cv2.imshow("data", data)
        # mask_red = cv2.inRange(src=data, lowerb=(0, 150, 60), upperb=(250, 240, 140))  # TERRACE RED
        # mask_yellow = cv2.inRange(src=data, lowerb=(0, 130, 0), upperb=(250, 200, 110)) # TERRACE YELLOW
        # mask_red = cv2.inRange(src=data, lowerb=(0, 180, 60), upperb=(250, 240, 140))  # CORY 337 RED
        # mask_yellow = cv2.inRange(src=data, lowerb=(0, 140, 0), upperb=(250, 200, 80))  # CORY 337 YELLOW
        mask_blue = cv2.inRange(src=data, lowerb=(60, 70, 120), upperb=(170, 130, 255))  # SHUWEI BLUE
        mask = mask_blue
        # mask = mask_red | mask_yellow

        # cv2.imshow("Lane Mask (Red)", mask_red)
        # cv2.imshow("Lane Mask (Yellow)", mask_yellow)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=1)
        cv2.imshow("Lane Mask (mask_blue)", mask)

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
