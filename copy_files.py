import os
import shutil

src = '/home/bo/Desktop/vmnih/data/mass_buildings/train/sat'
src_files = os.listdir(src)

dest_maps_list = os.listdir('/home/bo/Desktop/vmnih/data/mass_buildings_roads/train/map')

dest = '/home/bo/Desktop/vmnih/data/mass_buildings_roads/train/sat'

for file_name in dest_maps_list:
    full_file_name = os.path.join(src, file_name + 'f')
    if os.path.isfile(full_file_name):
        shutil.copy(full_file_name, dest)


