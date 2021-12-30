from pathlib import Path
from ROAR.utilities_module.data_structures_models import Transform, Location
import numpy as np
from typing import Tuple, Optional
import cv2
import plotly.express as px


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
        points = self.world_arr_to_occu_map(points)
        self.map[points[:, 1], points[:, 0]] = val
        return len(points)

    def visualize(self, dsize: Optional[Tuple] = None):
        img = self.map.copy()
        if dsize:
            img = cv2.resize(img, dsize=dsize)
        cv2.imshow(self.name, img)


waypoints = []
waypoints_arr = []
# debug waypoint following
f = Path("transforms_1.txt").open('r')
for line in f.readlines():
    x, y, z = line.split(",")
    x, y, z = float(x), float(y), float(z)
    l = Location(x=x, y=y, z=z)
    waypoints.append(Transform(location=l))
    waypoints_arr.append([x, z])
waypoints_arr = np.array(waypoints_arr)

buffer = 10
x_scale = 20
y_scale = 20

x_offset = abs(min(waypoints_arr[:, 0]))
y_offset = abs(min(waypoints_arr[:, 1]))

width = int((max(waypoints_arr[:, 0]) - min(waypoints_arr[:, 0])) * x_scale + x_offset + buffer)
height = int((max(waypoints_arr[:, 1]) - min(waypoints_arr[:, 1])) * y_scale + y_offset + buffer)
map = Map(x_offset=x_offset, y_offset=y_offset, x_scale=20, y_scale=20,
          x_width=width, y_height=height, buffer=buffer)
map.update(waypoints_arr)
map.visualize()
