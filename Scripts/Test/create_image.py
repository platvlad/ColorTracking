import cv2
import numpy as np

img = np.zeros(shape=(512, 640, 3))
img[:, :, 0] = 128
cv2.imwrite('/Users/vladislav.platonov/repo/RBOT2/RBOT/data/primitive/rgb/0001.png', img)
