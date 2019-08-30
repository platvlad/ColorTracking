import cv2
import os
import shutil

test_result_dir_name = 'D:\\results\\opt_tracking'
test_result_dir = os.fsencode(test_result_dir_name)

for directory in os.listdir(test_result_dir):
    directory_full = os.path.join(test_result_dir, directory)
    if os.path.isdir(directory_full):
        frames_dir = os.fsencode('output_frames')
        for test_case_directory in os.listdir(directory_full):
            test_case_directory_full = os.path.join(directory_full, test_case_directory)
            frames_dir_full = os.path.join(test_case_directory_full, frames_dir)
            frames_dir_name = os.fsdecode(frames_dir_full)

            cap = cv2.VideoCapture(frames_dir_name + '/%04d.png')
            capSize = (1920, 1280)
            fps = 30
            frame_width = int(cap.get(3))
            frame_height = int(cap.get(4))
            test_case_directory_name = os.fsdecode(test_case_directory)
            test_case_directory_full_name = os.fsdecode(test_case_directory_full)
            out = cv2.VideoWriter(test_case_directory_full_name + '/' + test_case_directory_name + '.avi',
                                  cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                                  25,
                                  (frame_width, frame_height))
            flag = True
            while cap.isOpened():
                ret, frame = cap.read()
                if frame is None:
                    break
                out.write(frame)
            shutil.rmtree(frames_dir_full)
            cap.release()
            out.release()
cv2.destroyAllWindows()
