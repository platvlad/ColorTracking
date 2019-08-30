import cv2
import os

file_name = \
    '/Users/vladislav.platonov/repo/RBOT2/RBOT/data/Vorona/src/Lis-B_55_2014-06-15_1401_C0001_Cheese_sh02_hd.mov'
video_file = os.fsencode(file_name)
video_dir_name = os.path.dirname(file_name)
vorona_dir_name = os.path.dirname(video_dir_name)
vorona_dir = os.fsencode(vorona_dir_name)
frames_dir = os.fsencode('rgb')
frames_dir_full = os.path.join(vorona_dir, frames_dir)
frames_dir_name = os.fsdecode(frames_dir_full)
os.makedirs(frames_dir_full)

cap = cv2.VideoCapture(file_name)
counter = 1
while cap.isOpened():
    frame_file_name = '000' + str(counter) + '.png' if counter < 10 \
        else '00' + str(counter) + '.png' if counter < 100 \
        else '0' + str(counter) + '.png' if counter < 1000 \
        else str(counter) + '.png'
    frame_file = os.fsencode(frame_file_name)
    frame_file_full = os.path.join(frames_dir_full, frame_file)
    frame_file_full_name = os.fsdecode(frame_file_full)
    ret, frame = cap.read()
    if frame is None:
        break
    cv2.imwrite(frame_file_full_name, frame)
    counter += 1
cap.release()
