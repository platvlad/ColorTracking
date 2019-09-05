import cv2
import os

first_to_save = 307
frames_dir_name = '/Users/vladislav.platonov/repo/RBOT2/RBOT/data/Vorona/rgb/'
frames_stream_name = frames_dir_name + '%04d.png'
frames_dir = os.path.dirname(frames_dir_name)

test_dir_name = os.path.dirname(frames_dir)
reduced_frames_dir_name = 'rgb' + str(first_to_save)
test_dir = os.fsencode(test_dir_name)
reduced_frames_dir = os.fsencode(reduced_frames_dir_name)
reduced_frame_dir_full = os.path.join(test_dir, reduced_frames_dir)
reduced_frame_dir_full_name = os.fsdecode(reduced_frame_dir_full)
os.makedirs(reduced_frame_dir_full)

cap = cv2.VideoCapture(frames_stream_name)
counter = 1
while True:
    ret, frame = cap.read()
    if frame is None:
        break
    if counter >= first_to_save:
        new_counter = counter - first_to_save + 1
        frame_file_name = '000' + str(new_counter) + '.png' if new_counter < 10 \
            else '00' + str(new_counter) + '.png' if new_counter < 100 \
            else '0' + str(new_counter) + '.png' if new_counter < 1000 \
            else str(new_counter) + '.png'
        frame_file = os.fsencode(frame_file_name)
        frame_file_full = os.path.join(reduced_frame_dir_full, frame_file)
        frame_file_full_name = os.fsdecode(frame_file_full)
        cv2.imwrite(frame_file_full_name, frame)
    counter += 1
cap.release()
