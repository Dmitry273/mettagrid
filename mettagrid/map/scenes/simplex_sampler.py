from typing import Any, NamedTuple
import hydra
from opensimplex import OpenSimplex

import numpy as np

from mettagrid.map.scene import Scene
from mettagrid.map.node import Node
from mettagrid.map.utils.random import MaybeSeed

import math

'''
This file contains a set of functions that can be used to generate different types of noise patterns for terrain generation.
The functions are designed to be used with the OpenSimplex noise generator, and they can be combined in various ways to create complex terrain features.
The functions take in parameters such as x and y coordinates, width and height of the terrain, and various other parameters that control the noise generation process.
'''

def fn0(x,y, width, height, 
        lx:float = 0.1, 
        ly:float = 0.1, 
        ) -> tuple[float, float]:
    # simple function to generate additional noise
    return (x*lx, y*ly)

def fn1(x,y, width, height, **args) -> tuple[float, float]:
    #currently unused
    octave = [0.1,0.1]
    xi = abs(x-0.5*y)*(x-0.5*width)*(y-0.5*height) * octave[0]
    yi = abs(x-0.5*y)*(x-0.5*width)*(y-0.5*height) * octave[1]

    return (xi, yi)

def fn2(x,y, width, height, P, **args) -> tuple[float, float]:
    # currently unused
    octave = [0.1,0.1]
    xi = (x-0.5*width)  * octave[0]
    yi = (y-0.5*height) * octave[1]

    a = (math.pi / 4 * (math.sqrt(xi ** 2 + yi ** 2) * (2+P)))
    xi, yi = xi * math.cos(a) - yi * math.sin(a), yi * math.cos(a) + xi * math.sin(a)

    return (xi, yi)

def fn3(x,y, width, height, **args) -> tuple[float, float]:
    # currently unused
    octave = [0.1,0.1]
    if x%7 == 0 or y%3 == 0:
        xi, yi = 1*x, 4*y
    else:
        xi, yi = -2*x, -5*y

    return (xi*octave[0], yi*octave[1])

def fn4(x, y, width, height,
        lx:float = 0.1,
        ly:float = 0.1, 
        t:float = 0.25, 
        x_pow:int = 2, 
        y_pow:int = 2, 
        xc:float = 0.0, 
        yc:float = 0.0, 
        P:float = 1.0, 
        ax:float = 0.0,
        ay:float = 0.0,
        bx:float = 0.0,
        by:float = 0.0,
        ) -> tuple[float, float]:
    # function used in "the_sphere" and "the_what" generator
    xi = (x-(0.5+xc)*width)*0.05
    yi = (y-(0.5+yc)*height)*0.05
    alpha = 2*math.pi*t
    cs = math.cos(alpha)
    sn = math.sin(alpha)
    xi, yi = (cs*(xi) + sn*(yi)), (-sn*(xi) + cs*(yi))
    a = np.sinc(((bx+xi)**x_pow+(by+yi)**y_pow)*math.sin(math.atan2((y-(0.5+ay)*height),(x-(0.5+ax)*width))*P))
    xi, yi = a, a

    return (xi*lx, yi*ly)

def fn5(x,y, width, height, M, **args) -> tuple[float, float]:
    # currently unused
    octave = [0.5,0.5]
    xi = (x-M*width)**2 * octave[0]
    yi = (y-0.5*height)**2 * octave[1]

    return (xi, yi)

def fn6(x,y, width, height, 
        lx:float = 0.1, 
        ly:float = 0.1, 
        t:float = 0.25, 
        x_pow:int = 2, 
        y_pow:int = 2, 
        xc:float = 0.0, 
        yc:float = 0.0, 
        ) -> tuple[float, float]:
    # function used in "cross_curse" generator
    alpha = 2*math.pi*t
    cs = math.cos(alpha)
    sn = math.sin(alpha)
    xi, yi = x, y

    xi, yi = (cs*(xi-(0.5+xc)*width) + sn*(yi-(0.5+yc)*height)), (-sn*(xi-(0.5+xc)*width) + cs*(yi-(0.5+yc)*height))

    xi, yi = 8 * xi**x_pow /x_pow**x_pow, 8 * yi**y_pow /y_pow**y_pow
 
    return (xi*lx, yi*ly)

def fn7(x,y, width, height, 
        lx:float = 0.1, 
        ly:float = 0.1, 
        t:float = 0.25, 
        symmetry:int = 3, 
        xc:float = 0.0, 
        yc:float = 0.0, 
        ) -> tuple[float, float]:
    # function used in "symmetry" generator
    alpha = 2*math.pi*t
    cs = math.cos(alpha)
    sn = math.sin(alpha)
    xi, yi = (cs*(x-0.5*width+xc*width) + sn*(y-0.5*height+yc*height)), (-sn*(x-0.5*width+xc*width) + cs*(y-0.5*height+yc*height))
    a = 0
    b = 0
    beta = (symmetry-1)*math.atan2((yi+b*height), (xi+a*width))
    csb = math.cos(beta)
    snb = math.sin(beta)
    xi, yi = (csb*(xi+a*width) - snb*(yi+b*height)), (snb*(xi+a*width) + csb*(yi+b*height))

    xi, yi = (cs*(xi-0.5*width+xc*width) - sn*(yi-0.5*height+yc*height)), (sn*(xi-0.5*width+xc*width) + cs*(yi-0.5*height+yc*height))
 
    return (xi*lx, yi*ly)

class Layer(NamedTuple):
    fn: 'function'
    saturation: float
    params: dict[str,Any]

class SimplexSampler(Scene):

    EMPTY, WALL = "empty", "wall"

    def __init__(
        self,
        room_size: int = 1,
        wall_size: int = 1,
        seed: MaybeSeed = None,
        children: list[Any] = [],
        layers: list[Layer] = [], # layers dictate how generated noise is sampled and how the end result will look
        cutoff: int = 70,
    ):
        super().__init__(children=children)
        self._room_size = room_size
        self._wall_size = wall_size
        self.seed = seed
        self.cutoff = cutoff
        self._rng = np.random.default_rng(seed)
        self.layers = [Layer(hydra.utils.get_method(x.fn), x.saturation, x.params) for x in layers if isinstance(x.fn, str)]
        
    def _render(self, node: Node) -> None:
        grid = node.grid
        self._height, self._width = node.grid.shape
        
        terrain = np.ones(shape=(grid.shape)) # template neutral terrain of ones to prevent errors
        for layer in self.layers:
            if abs(layer.saturation) > 0.00001: terrain *= self.terrain_map(layer)      # terrains layered on top of each other via multiplication,
                                                                                        # terrains with ~0 saturation are skipped from calculation
        terrain = self.normalize_array(terrain)                                         # sets min value as 0, max as 1 via appropriate rescaling
        terrain = (terrain * 255).astype('uint8')

        room = np.array(np.where(terrain > self.cutoff, self.EMPTY,self.WALL), dtype='<U50')
        self.fix_map(room)
        grid[:] = room
    
    def normalize_array(self, room: np.ndarray) -> np.ndarray:
        norm_arr = (room - np.min(room)) / (np.max(room) - np.min(room))
        return norm_arr
    
    def terrain_map(self, layer: Layer) -> np.ndarray:
        simplex = OpenSimplex(self._rng.integers(0, 2**31-1))

        xa = np.array(range(self._width))
        ya = np.array(range(self._height))
        terrain = np.array([simplex.noise2(*layer.fn(x, y, self._width, self._height, **layer.params)) for y in ya for x in xa]) # fn function dictates where and how fast noise will be sampled for each pixel in a room, absolute value is less important than derivative of this function
                                                                                                              # the faster noise is sampled: fn(x,y) ~ fn(x+1,y) in some region the less changes will be in this region per pixel, noise will look zoomed in
                                                                                                              # the slower noise is sampled: fn(x,y) !=fn(x+1,y) in some region the more changes will be in this region per pixel, noise will look zoomed out
        terrain = (terrain + 1)/2 # changes range from [-1,1] to [0,1]
        terrain = terrain.reshape((self._height,self._width))
        terrain = terrain**layer.saturation # saturates pattern with walls. Helpful since base noise is balanced to be 50/50. Saturation 0 makes neutral terrain with 0 walls, saturation >100 fills everything with walls.

        return terrain
    
    def fix_map(self, room: np.ndarray) -> None:
        #find any empty cell as start
        start = np.where(room == 'empty')
        if start[0].size == 0:
            return
        start = (start[0][0],start[1][0])

        def out_of_bound(y, x):
            if x < 0 or y < 0:
                return True
            if x >= self._width or y >= self._height:
                return True
            return False
        
        dir_list = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        while True:
            closed_set = {}
            open_list_empty = [start]
            open_list_wall = []

            #find area border
            while open_list_empty:
                current = open_list_empty.pop(0)
                if room[current[0], current[1]] == 'wall':
                    open_list_wall.append(current)
                    continue
                for next_dir in dir_list:
                    next_pos = (current[0] + next_dir[0], current[1] + next_dir[1])
                    if out_of_bound(next_pos[0], next_pos[1]):
                        continue
                    if next_pos in closed_set:
                        continue
                    closed_set[next_pos] = 1
                    open_list_empty.append(next_pos)

            #find another empty area
            predecessor_map = {}
            another_empty_cell = (-1, -1)
            while open_list_wall:
                current = open_list_wall.pop(0)
                if room[current[0], current[1]] == 'empty':
                    another_empty_cell = current
                    break
                for next_dir in dir_list:
                    next_pos = (current[0] + next_dir[0], current[1] + next_dir[1])
                    if out_of_bound(next_pos[0], next_pos[1]):
                        continue
                    if next_pos in closed_set:
                        continue
                    closed_set[next_pos] = 1
                    predecessor_map[next_pos] = current
                    open_list_wall.append(next_pos)

            #cannot find another empty area, break
            if another_empty_cell == (-1, -1):
                break

            #link two empty areas
            while another_empty_cell in predecessor_map:
                predecessor = predecessor_map[another_empty_cell]
                room[predecessor[0]][predecessor[1]] = 'empty'
                another_empty_cell = predecessor