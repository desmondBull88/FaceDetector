import cv2
import pandas
import time
from datetime import datetime
first_frame = None
list_stat = [None, None]
times = []
vid = cv2.VideoCapture(0)
df = pandas.DataFrame(columns=["Start", "End"])
while True:

    check, frame = vid.read()
    status = 0
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    if first_frame is None:
        first_frame = gray
        continue

    delta_frame = cv2.absdiff(first_frame, gray)

    threshFrame = cv2.threshold(delta_frame, 30, 255, cv2.THRESH_BINARY)[1]

    threshFrame = cv2.dilate(threshFrame, None, iterations=2)

    (cnts, _) = cv2.findContours(threshFrame.copy(),
                                 cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for countour in cnts:
        if cv2.contourArea(countour) < 10000:
            continue
        status = 1

        (x, y, w, h) = cv2.boundingRect(countour)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 3)
    list_stat.append(status)

    if list_stat[-1] == 1 and list_stat[-2] == 0:
        times.append(datetime.now())
    if list_stat[-1] == 0 and list_stat[-2] == 1:
        times.append(datetime.now())

    cv2.imshow("Capturing", frame)

    key = cv2.waitKey(1)

    if key == ord('q'):
        if status == 1:
            times.append(datetime.now())
        break

for i in range(0, len(times), 2):
    df = df.append({"Start": times[i], "End": times[i+1]}, ignore_index=True)

df.to_csv("Times.csv")
vid.release()
cv2.destroyAllWindows
