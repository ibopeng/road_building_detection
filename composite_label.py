"""
Combine road and building labels into one single image

Created on Sun March 17 2018
@ Author: Bo Peng
@ University of Wisconsin - Madison
@ Project: Road Extraction
"""

import os
from skimage.io import imread
from skimage.color import rgb2gray
import numpy as np
from scipy.misc import imsave

dirpath_bldg_map = '/home/bo/Desktop/vmnih/data/mass_buildings/train/map'
dirpath_road_map = '/home/bo/Desktop/vmnih/data/mass_roads/train/map'
dirpath_merge_map = '/home/bo/Desktop/vmnih/data/mass_buildings_roads/train/map'
dirpath_merge_map_binary = '/home/bo/Desktop/vmnih/data/mass_buildings_roads/train/map_binary'

# Note that all building images are contained in road set
# get file name
filelist_bldg = os.listdir(dirpath_bldg_map)
filelist_road = os.listdir(dirpath_road_map)

filelist_bldg_road = []
# check which bldg file is in road set
for fb in filelist_bldg:
    if fb in filelist_road:
        filelist_bldg_road.append(fb)

filepath_bldg_map = []
filepath_road_map = []
filepath_merge_map = []
filepath_merge_map_binary = []

for i in filelist_bldg_road:
    filepath_bldg_map.append(os.path.join(dirpath_bldg_map, i))
    filepath_road_map.append(os.path.join(dirpath_road_map, i))
    filepath_merge_map.append(os.path.join(dirpath_merge_map, i))
    filepath_merge_map_binary.append(os.path.join(dirpath_merge_map_binary, i[:-4]))

# read maps from road set and bldg set
map_bldg = [rgb2gray(imread(fb)) for fb in filepath_bldg_map]
map_road = [imread(fr) for fr in filepath_road_map]

# create new maps containing bldg and road
for i in range(len(filelist_bldg_road)):
    fp_merge = filepath_merge_map[i]
    fp_merge_binary = filepath_merge_map_binary[i]
    # create a new map [1500, 1500]
    map_merge = np.zeros((1500, 1500), dtype=np.int32)

    # check both maps
    mb = map_bldg[i]
    mr = map_road[i]
    for j in range(1500):
        for k in range(1500):
            if mb[j][k] > 0:  # building pixels
                map_merge[j][k] = 1
            elif mr[j][k] > 0:  # road pixels
                map_merge[j][k] = 2

    # save this merged map
    imsave(fp_merge, map_merge)
    # save the merged map into binary file
    np.save(fp_merge_binary, map_merge)


print('done')




