from skimage.io import imread
import numpy as np

# imb = np.load('/home/bo/Desktop/vmnih/data/mass_buildings_roads/train/map_binary/22679050_15.npy')

im = imread('./mass_buildings_roads/train/sat/22978870_15.tiff')
im = np.array(im[400:900][500:1000])

im_patch = im[0][0]

b = all(im_patch == [254, 255, 255])

print('done')