import cv2
import time

first_frame = None

vid = cv2.VideoCapture(0)

while True:

    check, frame = vid.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    if first_frame is None:
        first_frame = gray
        continue

    delta_frame = cv2.absdiff(first_frame, gray)

    threshFrame = cv2.threshold(delta_frame, 30, 255, cv2.THRESH_BINARY)[1]

    threshFrame = cv2.dilate(threshFrame, None, iterations=2)

    cv2.imshow("Capturing", frame)

    key = cv2.waitKey(1)

    if key == ord('q'):
        break
vid.release()
cv2.destroyAllWindows
