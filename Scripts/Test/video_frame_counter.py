import cv2
import time
cap = cv2.VideoCapture('/Users/vladislav.platonov/repo/RBOT2/RBOT/data/fox_head/fox_head_2/rgb.mov')

flag = True
counter = 0
while cap.isOpened():
    ret, frame = cap.read()
    if frame is None:
        break
    else:
        counter += 1

print(counter)
cap.release()
cv2.destroyAllWindows()
