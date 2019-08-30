import cv2
import numpy as np

mask = cv2.imread('/Users/vladislav.platonov/repo/ColorTracking/ColorTracking/data/cube-3/mask.png')
img = cv2.imread('/Users/vladislav.platonov/repo/ColorTracking/ColorTracking/data/ir_ir_5_r/rgb/0001.png')
x = 887
y = 372
width = 132
height = 275

#
# real_high_y = 1120 - low_y
# real_low_y = 1120 - high_y
#
# epsilon = 250
# interesting_x = 754
# interesting_y = 746
#
roi = img[y:y + height, x:x + width]

# for i in range(len(mask)):
#     for j in range(len(mask[i])):
#         if mask[i][j][0] and mask[i][j][2] and i != 745:
#             mask[i][j][0] = 0
#             mask[i][j][1] = 0
#             mask[i][j][2] = 255
#         if mask[i][j][0] and mask[i][j][2] and i == 745:
#             mask[i][j][0] = 255
#             mask[i][j][1] = 0
#             mask[i][j][2] = 0


# roi = mask[interesting_y - epsilon:interesting_y + epsilon, interesting_x - epsilon:interesting_x + epsilon, :]
# for i in range(len(mask)):
#     for j in range(len(mask[i])):
#         if mask[i][j][0]:
#             mask[i][j][0] = 128
#             mask[i][j][1] = 128
#             mask[i][j][2] = 128
#             img[i][j][0] = 0
#             img[i][j][1] = 0
#             img[i][j][2] = 255
#         else:
#             img[i][j][0] = 255
#             img[i][j][1] = 0
#             img[i][j][2] = 0
# print(a)
cv2.imshow('blue_with_gray', roi)
cv2.waitKey()
# cv2.imwrite("/Users/vladislav.platonov/repo/ColorTracking/ColorTracking/data/cube-3/rgb/0002.png", img)
