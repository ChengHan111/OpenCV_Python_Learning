import matplotlib.pyplot as plt
import cv2 as cv
structure = cv.imread('WeChat Image_20210320120051.png')
plt.imshow(structure)
plt.show()
# 在OpenCV中，我们要获取一个视频，需要创建一个VideoCapture对象
# 创建读取视频的对象
# cap = cv.VideoCapture()

# 获取视频的某些属性 retval = cap.get(propId)

# 修改视频的属性信息 cap.set(propId, value)

# 判断图像是否读取成功 isornot = cap.isOpened()

# 获取视频的一帧图像 ret, frame = cap.read() ret判断获取是否成功，成功为True frame为获取到的某一帧的图像
# 调用cv.imshow()显示图像，在显示图像时使用cv.waitKey()设置适当的持续时间，如果太低视频会播放的非常快，如果太高会非常慢，
# 通常情况设定为25ms即可
# 最后用cap.release()将视频释放掉

cap = cv.VideoCapture('Golden_Gate_Bridge__SaveYouTube_com_.mp4')
retval = cap.get(propId=3)
print(retval)
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        cv.imshow('frame', frame)
    if cv.waitKey(25) & 0xFF == ord('q'):
        break
cap.release()
cv.destroyAllWindows()

# 视频保存
# out = cv2.VideoWriter(filename,fourcc, fps, frameSize)
# filename：视频保存的位置
# fourcc：指定视频编解码器的4字节代码
# fps：帧率
# frameSize：帧大小
# 其中，fourcc是需要设置的，设置视频的编解码 retval = cv2.VideoWriter_fourcc( c1, c2, c3, c4 )
# c1,c2,c3,c4: 是视频编解码器的4字节代码，在fourcc.org中找到可用代码列表，与平台紧密相关，常用的有：
# In windows DIVX(.avi) In OS MJPG(.mp4), X264(.mkv) DIVX(.avi)
# 利用cap.read()获取每一帧，然后out.write()将某一帧图像写入视频中
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

out = cv.VideoWriter('Out.avi', cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame_width, frame_height))
while(True):
    ret, frame = cap.read()
    if ret == True:
        out.write(frame)
    else:
        break
cap.release()
out.release()
cv.destroyAllWindows()

# 读取视频：
#
#     读取视频：cap = cv.VideoCapture()
#     判断读取成功：cap.isOpened()
#     读取每一帧图像：ret,frame = cap.read()
#     获取属性：cap.get(proid)
#     设置属性：cap.set(proid,value)
#     资源释放：cap.release()
# 保存视频
#     保存视频： out = cv.VideoWrite()
#     视频写入：out.write()
#     资源释放：out.release()