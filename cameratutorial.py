import cv2 as cv
import numpy as np
cap = cv.VideoCapture(0)
while(True):
    _, frame = cap.read()
    # Convert BGR to HSV
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    # define range of blue color in HSV
    lower_blue = np.array([90,30,30])
    upper_blue = np.array([130,255,255])
    # Threshold the HSV image to get only blue colors
    mask = cv.inRange(hsv, lower_blue, upper_blue)
    # Bitwise-AND mask and original image
    res = cv.bitwise_and(frame,frame, mask= mask)
    try:
        contours,hierarchy = cv.findContours(mask, 1, 2)
        c = max(contours, key = cv.contourArea)


        x,y,w,h = cv.boundingRect(c)
        cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
    except:
        print("test")
        continue

    cv.imshow('frame',frame)
    # cv.imshow('mask',mask)
    # cv.imshow('res',res)
    k = cv.waitKey(5) & 0xFF
    if k == 27:
        break
cv.destroyAllWindows()

