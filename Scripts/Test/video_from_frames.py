import cv2

directory_name = '/Users/vladislav.platonov/repo/ColorTracking/ColorTracking/data/ir_ir_5_r/mead5'
output_video_name = directory_name + '/output.avi'

cap = cv2.VideoCapture(directory_name + '/%04d.png')

capSize = (1920, 1280)
fps = 30
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

out = cv2.VideoWriter(output_video_name, cv2.VideoWriter_fourcc('M','J','P','G'), 25, (frame_width, frame_height))

flag = True
while cap.isOpened():
    ret, frame = cap.read()

    if frame is None:
        break

    # if ret == True:
    #     frame = cv2.flip(frame, 0)

    out.write(frame)

    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #
    # cv2.imshow('frame', frame)
    # if flag:
    #     cv2.waitKey(30000)
    #     flag = False
    # cv2.waitKey(1)
    # if cv2.waitKey(30) & 0xFF == ord('q'):
    #     break

cap.release()
out.release()
cv2.destroyAllWindows()
