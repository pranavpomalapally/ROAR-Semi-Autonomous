from typing import Tuple, Optional
import open3d as o3d
import os
from glob import glob
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt
from ROAR.utilities_module.data_structures_models import Location
from tqdm import tqdm


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

    def occu_map_to_world(self, point: Tuple[int, int]) -> Tuple[float, float]:
        x, y = point
        X = (x - self.buffer) / self.x_scale - self.x_offset
        Y = (y - self.buffer) / self.y_scale - self.y_offset
        return X, Y

    def get_all_obstacles_in_world(self, threshold=0.5) -> Tuple[np.ndarray, np.ndarray]:
        xs, ys = np.where(self.map > threshold)
        return (xs - self.buffer) / self.x_scale - self.x_offset, (ys - self.buffer) / self.y_scale - self.y_offset

    def pcd_to_occu_map(self, pcd: o3d.geometry.PointCloud):
        points = np.asarray(pcd.points)
        xs = np.expand_dims(points[:, 0], axis=1)
        zs = np.expand_dims(points[:, 2], axis=1)
        arr = np.hstack([xs, zs])
        return self.world_arr_to_occu_map(arr)

    @staticmethod
    def extract_xz_points_from_pcd(pcd: o3d.geometry.PointCloud) -> np.ndarray:
        points = np.asarray(pcd.points)
        xs = np.expand_dims(points[:, 0], axis=1)
        zs = np.expand_dims(points[:, 2], axis=1)
        points = np.hstack([xs, zs])
        return points

    def update(self, points: np.ndarray, val=1) -> int:
        """

        Args:
            val: value to update those points to
            points: points is a 2D numpy array consist of X and Z coordinates

        Returns:
            number of points updated
        """
        points = self.world_arr_to_occu_map(points)
        self.map[points[:, 1], points[:, 0]] = val
        return len(points)

    def visualize(self, dsize: Optional[Tuple] = None):
        img = self.map.copy()
        if dsize:
            img = cv2.resize(img, dsize=dsize)
        plt.imshow(img, cmap='gray')
        plt.show()
        # cv2.imshow(self.name, img)

    @staticmethod
    def moving_average(a, n=3):
        ret_x = np.cumsum(a[:, 0], dtype=float)
        ret_z = np.cumsum(a[:, 1], dtype=float)

        ret_x[n:] = ret_x[n:] - ret_x[:-n]
        ret_z[n:] = ret_z[n:] - ret_z[:-n]

        ret_x = ret_x[n - 1:] / n
        ret_z = ret_z[n - 1:] / n

        return np.array([ret_x, ret_z]).T

    @staticmethod
    def find_hyperparam_from_pcd(pcd: o3d.geometry.PointCloud, scale, buffer):
        x_offset = np.max([abs(np.floor(pcd.get_min_bound()[0])), abs(np.ceil(pcd.get_max_bound()[0]))]).astype(int)
        y_offset = np.max([abs(np.floor(pcd.get_min_bound()[2])), abs(np.ceil(pcd.get_max_bound()[2]))]).astype(int)
        x_width = np.ceil((pcd.get_max_bound()[0] + x_offset) * scale + buffer).astype(int)
        y_height = np.ceil((pcd.get_max_bound()[2] + y_offset) * scale + buffer).astype(int)
        return x_offset, y_offset, x_width, y_height

    @staticmethod
    def read_pointclouds(dir_path: Path, voxel_down_sample=0.1) -> o3d.geometry.PointCloud:
        assert dir_path.exists(), f"{dir_path} does not exist"
        # load the files
        pointcloud_folder = Path(dir_path)
        files = glob(pointcloud_folder.as_posix() + "/*.pcd")
        files.sort(key=os.path.getmtime)
        pcd = o3d.geometry.PointCloud()
        for file in tqdm(files):
            p = o3d.io.read_point_cloud(file)
            p_points = np.asarray(p.points)
            p_colors = np.asarray(p.colors)

            points = np.concatenate((np.asarray(pcd.points), p_points), axis=0)
            colors = np.concatenate((np.asarray(pcd.colors), p_colors), axis=0)

            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors)

            pcd = pcd.voxel_down_sample(voxel_down_sample)
        return pcd


if __name__ == "__main__":
    import socket
    import time

    HOST = '192.168.1.32'  # The server's hostname or IP address
    PORT = 80  # The port used by the server
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((HOST, PORT))
    # for i in range(10):
    i = 2048
    msg = f"T{i}"
    s.send(msg.encode('utf-8'))
    print(f"sent: {msg}")
    time.sleep(0.1)
