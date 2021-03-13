import numpy as np
import cv2 as cv

# 1 获取视频
cap = cv.VideoCapture('sample-wmv-file.wmv')

# 2 指定追踪目标
ret,frame = cap.read()
r,h,c,w=100,30,100,30
win = (c,r,w,h)
roi = frame[r:r+h,c:c+w]

# 3 计算直方图
hsv_roi = cv.cvtColor(roi,cv.COLOR_BGR2HSV)
roi_hist = cv.calcHist([hsv_roi],[0],None,[180],[0,180])
cv.normalize(roi_hist,roi_hist,0,255,cv.NORM_MINMAX)

# 4 目标追踪
term = (cv.TERM_CRITERIA_EPS|cv.TERM_CRITERIA_COUNT,10,1)

while(True):
    ret,frame = cap.read()
    if ret ==True:
        hst = cv.cvtColor(frame,cv.COLOR_BGR2HSV)
        dst = cv.calcBackProject([hst],[0],roi_hist,[0,180],1)
        ret,win = cv.CamShift(dst,win,term)

        pts = cv.boxPoints(ret)
        pts = np.int0(pts)

        img2 = cv.polylines(frame, [pts], True, 255,2)
        cv.imshow("frame",img2)
        if cv.waitKey(60)&0xFF ==ord('q'):
            break


# 5 释放资源
cap.release()
cv.destroyAllWindows()