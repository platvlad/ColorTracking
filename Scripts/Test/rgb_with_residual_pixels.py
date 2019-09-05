import cv2
import numpy as np

file_name = '/Users/vladislav.platonov/repo/RBOT2/RBOT/data/opt_tracking_small/opt_tracking/ironman/ir_ir_5_r/errors2.1.txt'
x = []
y = []
with open(file_name) as coords_data:
    coords_list = coords_data.readlines()
    for coords in coords_list:
        x.append(int(coords[4:7]))
        y.append(int(coords[13:16]))

rgb = cv2.imread('/Users/vladislav.platonov/repo/RBOT2/RBOT/data/opt_tracking_small/opt_tracking/ironman/ir_ir_5_r/rgb/0002.png')
rgb_old = np.copy(rgb)

mask = cv2.imread('/Users/vladislav.platonov/repo/RBOT2/RBOT/data/opt_tracking_small/opt_tracking/ironman/ir_ir_5_r/mask2.png')

for i in range(len(x)):
    mask_px = mask[y[i]][x[i]]
    if mask_px[0] or mask_px[1] or mask_px[2]:
        rgb[y[i]][x[i]] = [0, 255, 0]
    else:
        rgb[y[i]][x[i]] = [255, 0, 0]
cv2.imshow('ironman', rgb)
cv2.waitKey()
cv2.imshow('rgb_old', rgb_old)
cv2.waitKey()