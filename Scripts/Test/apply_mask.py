import cv2
import numpy as np

mask = cv2.imread('/Users/vladislav.platonov/repo/RBOT2/RBOT/data/opt_tracking_small/opt_tracking/ironman/ir_ir_5_r/mask2.png')
rgb = cv2.imread('/Users/vladislav.platonov/repo/RBOT2/RBOT/data/opt_tracking_small/opt_tracking/ironman/ir_ir_5_r/rgb/0002.png')

centers_file = '/Users/vladislav.platonov/repo/RBOT2/RBOT/data/opt_tracking_small/opt_tracking/ironman/ir_ir_5_r/centers.txt'
with open(centers_file) as centers_data:
    center_lines = centers_data.readlines()
    center_coords = [line.split() for line in center_lines]


for i in range(len(rgb)):
    for j in range(len(rgb[i])):
        if not mask[i][j][0] and not mask[i][j][1] and not mask[i][j][2]:
            # rgb[i][j] = [0, 0, 0]
            rgb[i][j][0] = rgb[i][j][0] / 2
            rgb[i][j][1] = rgb[i][j][1] / 2
            rgb[i][j][2] = rgb[i][j][2] / 2
for center in center_coords:
    i, j, id = center
    i = int(i)
    j = int(j)
    id = int(id)
    rgb[j][i] = [0, 255, 0]
cv2.imshow('rgb masked', rgb)
cv2.waitKey()
