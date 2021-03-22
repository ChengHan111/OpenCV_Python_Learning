# Harr 用以人脸特征检测，每一个特征是一个值，这个值等于黑色举行中的像素值之和减去白色矩形区域中的像素值之和
# Harr特征反映了图像的灰度变换情况，可以用于图像任意位置，大小也可以任意变换
# 得到图像特征之后，训练一个决策树构建的adaboost级联决策器来识别是否为人脸
# Opencv中自带了训练好的检测器，包括面部，眼睛，猫脸等，都保存在xml中
import cv2 as cv
print(cv.__file__)
# 流程：
# 1.读取图片，将其转换成为灰度图
# 2.实例化人脸和眼睛检测的分类器对象
# 2.1 实例化级联分类器 classifier = cv.CascadeClassifier("haarcascade_frontalface_default.xml")
# 2.2 加载分类器 classifier.load("haarcascade_frontalface_default.xml")
# 3.进行人脸和眼睛的检测
# 3.1 rect = classifier.detectMultiScale(gray, scaleFactor, minNeighbors, minSize, maxSize)
# Gray：要进行检测的人脸图像
# scaleFactor：前后两次扫面中，搜索窗口的比例系数
# minNeighbors：目标至少被检测到minNeighbors次才会被认为是目标
# minSize和maxSize：目标的最小尺寸和最大尺寸

import matplotlib.pyplot as plt
img = cv.imread('wulin.jpeg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

face_cas = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
face_cas.load('haarcascade_frontalface_default.xml')

eyes_cas = cv.CascadeClassifier('haarcascade_eye.xml')
eyes_cas.load('haarcascade_eye.xml')

faceRects = face_cas.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(23,23))
for faceRect in faceRects:
    x,y,w,h = faceRect
    # 框出人脸
    cv.rectangle(img, (x,y), (x + w, y + h), (0,255,0), 3)
    # 再框出人脸之后再进行眼睛检测
    '''注意这里是先圈的y再圈的x, roi_interest'''
    # 这是因为在opencv中，img协调序列是y坐标，然后是x坐标!!!! Important
    roi_color = img[y:y+h, x:x+w]
    roi_gray = gray[y:y + h, x:x + w]
    eyes = eyes_cas.detectMultiScale(roi_gray)
    for (ex, ey, ew, eh) in eyes:
        cv.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0,255,0), 2)
plt.figure(figsize=(8,6),dpi=100)
plt.imshow(img[:,:,::-1])
plt.show()

'''
在视频中的应用
'''
cap = cv.VideoCapture('video.mp4')
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        face_cas = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
        face_cas.load('haarcascade_frontalface_default.xml')
        eyes_cas = cv.CascadeClassifier('haarcascade_eye.xml')
        eyes_cas.load('haarcascade_eye.xml')
        faceRects = face_cas.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(100, 100))
        for faceRect in faceRects:
            x, y, w, h = faceRect
            cv.rectangle(frame, (x,y), (x + w, y + h), (0, 255, 0), 2)
            roi_color = frame[y:y+h, x:x+w]
            roi_gray = gray[y:y+h, x:x+w]
            eyes = eyes_cas.detectMultiScale(roi_gray,scaleFactor=1.2, minNeighbors=2, minSize=(80,80))
            for (ex, ey, ew, eh) in eyes:
                cv.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 0, 255), 2)

        cv.imshow('frame', frame)
        if cv.waitKey(1) & 0XFF == ord('q'):
            break
cap.release()
cv.destroyAllWindows()
