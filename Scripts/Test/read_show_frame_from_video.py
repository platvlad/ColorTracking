import cv2

cap = cv2.VideoCapture('/Users/vladislav.platonov/repo/RBOT2/RBOT/data/primitive/output.mov')
while cap.isOpened():
    ret, frame = cap.read()
    cv2.imshow('frame', frame)
    cv2.waitKey()
    break
