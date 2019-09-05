import cv2
import numpy as np

# img = cv2.imread('/Users/vladislav.platonov/repo/ColorTracking/ColorTracking/cmake-build-debug/depth.png', cv2.IMREAD_GRAYSCALE)
# int_sdt = cv2.imread('/Users/vladislav.platonov/repo/ColorTracking/ColorTracking/cmake-build-debug/internal_signed_distance.png', cv2.IMREAD_ANYDEPTH)
# sdt = cv2.imread('/Users/vladislav.platonov/repo/ColorTracking/ColorTracking/cmake-build-debug/signed_distance.png', cv2.IMREAD_ANYDEPTH)
# sdt_rows = cv2.imread('/Users/vladislav.platonov/repo/RBOT2/RBOT/data/primitive/sdt_rows.png', cv2.IMREAD_GRAYSCALE)
# mask = cv2.imread('/Users/vladislav.platonov/repo/ColorTracking/ColorTracking/cmake-build-debug/mask.png', cv2.IMREAD_ANYDEPTH)
depth = cv2.imread('/Users/vladislav.platonov/repo/ColorTracking/ColorTracking/cmake-build-debug/depth.png', cv2.IMREAD_GRAYSCALE)
old_depth = cv2.imread('/Users/vladislav.platonov/repo/ColorTracking/ColorTracking/cmake-build-debug/old_depth.png', cv2.IMREAD_GRAYSCALE)
# dst = cv2.distanceTransform(mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
cv2.imshow('depth', depth)
cv2.imshow('old_depth', old_depth)
cv2.waitKey()
# img[:] *= 255
# for row in img:
#     for column in row:
#         for ch in column:
#             if ch > 0:
#                 print(ch)
# for i in range(len(img)):
#     for j in range(len(img[0])):
#         px = img[i][j]
#         if px[0] == 64 and px[1] == 69 and px[2] == 98:
#             print(i, j)
#
# cv2.imshow("ironman", img)
# cv2.waitKey()
