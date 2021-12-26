import json

from ROAR.control_module.controller import Controller
from ROAR.utilities_module.data_structures_models import Transform
from ROAR.utilities_module.vehicle_models import Vehicle
from ROAR.utilities_module.vehicle_models import VehicleControl
from collections import deque
import numpy as np
from ROAR_iOS.config_model import iOSConfig
from pathlib import Path


class IntelRacingImagePIDController(Controller):
    def __init__(self, agent, **kwargs):
        super().__init__(agent, **kwargs)
        self.long_error_deque_length = 50
        self.lat_error_deque_length = 10
        self.lat_error_queue = deque(
            maxlen=self.lat_error_deque_length)  # how much error you want to accumulate
        self.long_error_queue = deque(
            maxlen=self.long_error_deque_length)  # how much error you want to accumulate
        self.target_speed = 7  # target speed that you want to pursue at. m / s
        self.max_throttle = 0.16
        self.curr_max_throttle = self.max_throttle
        self.config = json.load(Path(self.agent.agent_settings.pid_config_file_path).open('r'))
        self.long_config = self.config["longitudinal_controller"]
        self.lat_config = self.config["latitudinal_controller"]

    def run_in_series(self, next_waypoint=None, **kwargs) -> VehicleControl:
        steering = self.lateral_pid_control()
        throttle = self.long_pid_control()
        return VehicleControl(throttle=throttle, steering=steering)

    def lateral_pid_control(self) -> float:
        error = self.agent.kwargs.get("lat_error", 0)
        error_dt = 0 if len(self.lat_error_queue) == 0 else error - self.lat_error_queue[-1]
        self.lat_error_queue.append(error)
        error_it = sum(self.lat_error_queue)
        k_p, k_d, k_i = self.find_k_values(self.agent.vehicle, self.lat_config)
        # print(f"Speed = {self.agent.vehicle.get_speed(self.agent.vehicle)}"
        #       f"kp, kd, ki = {k_p, k_d, k_i} ")
        e_p = k_p * error
        e_d = k_d * error_dt
        e_i = k_i * error_it
        lat_control = np.clip((e_p + e_d + e_i), -1, 1)
        # print(f"speed = {self.agent.vehicle.get_speed(self.agent.vehicle)} "
        #       f"e = {round((e_p + e_d + e_i), 3)}, "
        #       f"e_p={round(e_p, 3)},"
        #       f"e_d={round(e_d, 3)},"
        #       f"e_i={round(e_i, 3)},"
        #       f"lat_control={lat_control}")
        # print()
        return lat_control

    def long_pid_control(self) -> float:
        k_p, k_d, k_i = self.find_k_values(self.agent.vehicle, self.long_config)
        e = self.target_speed - self.agent.vehicle.get_speed(self.agent.vehicle)
        neutral = -90
        incline = self.agent.vehicle.transform.rotation.pitch - neutral

        # e = e * - 1 if incline < -10 else e
        self.long_error_queue.append(e)
        de = 0 if len(self.long_error_queue) < 2 else self.long_error_queue[-2] - self.long_error_queue[-1]
        ie = 0 if len(self.long_error_queue) < 2 else np.sum(self.long_error_queue)
        incline = np.clip(incline, -20, 20)

        e_p = k_p * e
        e_d = k_d * de
        e_i = k_i * ie
        e_incline = 0.015 * incline

        total_error = e_p + e_d + e_i + e_incline
        # print(e_p, e_d, e_i, e_incline, total_error)

        if sum(self.long_error_queue) >= self.target_speed * (self.long_error_deque_length - 1):
            # if i have filled integral buffer with maximum amount of error, 
            # this probably means that i am unable to move forward with the given throttle
            # therefore, gradually increase the allowable throttle
            self.curr_max_throttle += 0.001
        else:
            self.curr_max_throttle = self.max_throttle

        if incline > 10:
            # if up ramp, execute a relatively big forward throttle to enable the vehicle to go forward
            self.curr_max_throttle = max(self.curr_max_throttle, 0.26)
        long_control = float(np.clip(total_error, -self.curr_max_throttle, self.curr_max_throttle))

        if incline < -10:
            # if downslope, execute a constant reverse throttle to counter the downward force
            p = -0.1
            self.logger.info(f"USING DOWNHILL CONSTANT P = {p} CONTROLLER")
            long_control = p

        return long_control

    @staticmethod
    def find_k_values(vehicle: Vehicle, config: dict) -> np.array:
        current_speed = Vehicle.get_speed(vehicle=vehicle)
        k_p, k_d, k_i = 1, 0, 0
        for speed_upper_bound, kvalues in config.items():
            speed_upper_bound = float(speed_upper_bound)
            if current_speed < speed_upper_bound:
                k_p, k_d, k_i = kvalues["Kp"], kvalues["Kd"], kvalues["Ki"]
                break
        return np.array([k_p, k_d, k_i])


class IntelRacingWaypointPIDController(Controller):
    def __init__(self, agent, **kwargs):
        super().__init__(agent, **kwargs)
        self.long_error_deque_length = 50
        self.lat_error_deque_length = 10
        self.lat_error_queue = deque(
            maxlen=self.lat_error_deque_length)  # this is how much error you want to accumulate
        self.long_error_queue = deque(
            maxlen=self.long_error_deque_length)  # this is how much error you want to accumulate
        self.target_speed = 7  # m / s
        self.max_throttle = 0.16
        self.curr_max_throttle = self.max_throttle
        self.config = json.load(Path(self.agent.agent_settings.pid_config_file_path).open('r'))
        self.long_config = self.config["longitudinal_controller"]
        self.lat_config = self.config["latitudinal_controller"]

    def run_in_series(self, next_waypoint=None, **kwargs) -> VehicleControl:
        steering = self.lateral_pid_control(next_waypoint)
        throttle = self.long_pid_control()
        return VehicleControl(throttle=throttle, steering=steering)

    def lateral_pid_control(self, next_waypoint:Transform) -> float:
        # calculate a vector that represent where you are going
        v_begin = self.agent.vehicle.transform.location.to_array()
        direction_vector = np.array([-np.sin(np.deg2rad(self.agent.vehicle.transform.rotation.yaw)),
                                     0,
                                     -np.cos(np.deg2rad(self.agent.vehicle.transform.rotation.yaw))])
        v_end = v_begin + direction_vector

        v_vec = np.array([(v_end[0] - v_begin[0]), 0, (v_end[2] - v_begin[2])])
        # calculate error projection
        w_vec = np.array(
            [
                next_waypoint.location.x - v_begin[0],
                0,
                next_waypoint.location.z - v_begin[2],
            ]
        )

        v_vec_normed = v_vec / np.linalg.norm(v_vec)
        w_vec_normed = w_vec / np.linalg.norm(w_vec)
        error = np.arccos(v_vec_normed @ w_vec_normed.T)
        _cross = np.cross(v_vec_normed, w_vec_normed)

        if _cross[1] > 0:
            error *= -1
        self.lat_error_queue.append(error)
        if len(self.lat_error_queue) >= 2:
            _de = (self.lat_error_queue[-1] - self.lat_error_queue[-2])
            _ie = sum(self.lat_error_queue)
        else:
            _de = 0.0
            _ie = 0.0

        k_p, k_d, k_i = self.find_k_values(config=self.lat_config, vehicle=self.agent.vehicle)

        lat_control = float(
            np.clip((k_p * error) + (k_d * _de) + (k_i * _ie), -1, 1)
        )
        return lat_control

    def long_pid_control(self) -> float:
        k_p, k_d, k_i = self.find_k_values(self.agent.vehicle, self.long_config)
        e = self.target_speed - self.agent.vehicle.get_speed(self.agent.vehicle)
        neutral = -90
        incline = self.agent.vehicle.transform.rotation.pitch - neutral

        # e = e * - 1 if incline < -10 else e
        self.long_error_queue.append(e)
        de = 0 if len(self.long_error_queue) < 2 else self.long_error_queue[-2] - self.long_error_queue[-1]
        ie = 0 if len(self.long_error_queue) < 2 else np.sum(self.long_error_queue)
        incline = np.clip(incline, -20, 20)

        e_p = k_p * e
        e_d = k_d * de
        e_i = k_i * ie
        e_incline = 0.015 * incline

        total_error = e_p + e_d + e_i + e_incline
        # print(e_p, e_d, e_i, e_incline, total_error)

        if sum(self.long_error_queue) >= self.target_speed * (self.long_error_deque_length - 1):
            self.curr_max_throttle += 0.001
        else:
            self.curr_max_throttle = self.max_throttle

        if incline > 10:
            self.curr_max_throttle = max(self.curr_max_throttle, 0.26)
        long_control = float(np.clip(total_error, -self.curr_max_throttle, self.curr_max_throttle))

        if incline < -10:
            p = -0.1
            self.logger.info(f"USING DOWNHILL CONSTANT P = {p} CONTROLLER")
            long_control = p

        return long_control

    @staticmethod
    def find_k_values(vehicle: Vehicle, config: dict) -> np.array:
        current_speed = Vehicle.get_speed(vehicle=vehicle)
        k_p, k_d, k_i = 1, 0, 0
        for speed_upper_bound, kvalues in config.items():
            speed_upper_bound = float(speed_upper_bound)
            if current_speed < speed_upper_bound:
                k_p, k_d, k_i = kvalues["Kp"], kvalues["Kd"], kvalues["Ki"]
                break
        return np.array([k_p, k_d, k_i])
