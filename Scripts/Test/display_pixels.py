import cv2
import numpy as np
bg_file_name = \
    '/Users/vladislav.platonov/repo/RBOT2/RBOT/data/opt_tracking_small/opt_tracking/ironman/ir_ir_5_r/bg_pixels.txt'
fg_file_name = \
    '/Users/vladislav.platonov/repo/RBOT2/RBOT/data/opt_tracking_small/opt_tracking/ironman/ir_ir_5_r/fg_pixels.txt'
with open(bg_file_name) as bg_file:
    bg_lines = bg_file.readlines()
    print(len(bg_lines))
with open(fg_file_name) as fg_file:
    fg_lines = fg_file.readlines()
    print(len(fg_lines))


def make_image(pixel_lines, size):
    pixels = np.zeros(shape=(size, size, 3))
    i = 0
    j = 0
    for pixel_line in pixel_lines:
        red, green, blue = pixel_line.split()
        r = int(red)
        g = int(green)
        b = int(blue)
        pixels[i][j] = [r / 256, g / 256, b / 256]
        j += 1
        if j == size:
            j = 0
            i += 1

    cv2.imshow('pixels', pixels)
    cv2.waitKey()

make_image(fg_lines, 448)
make_image(bg_lines, 1230)
