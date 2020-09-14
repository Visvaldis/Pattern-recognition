import numpy as np
import cv2

cap = cv2.VideoCapture(0)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))

while(cap.isOpened()):
    ret, frame = cap.read()
    
    if ret==True:
        frame = cv2.flip(frame, 90)

        out.write(frame)
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()

capFile = cv2.VideoCapture('output.avi')

while(capFile.isOpened()):
    ret, frame = capFile.read()

    gray = cv2.cvtColor(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)

    cv2.rectangle(gray, (0, 0), (130, 140), (0, 255, 0), 2)
    cv2.line(gray, (0, 0), (50, 100), (0, 0, 255), 2)

    cv2.imshow('frame', gray)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()


