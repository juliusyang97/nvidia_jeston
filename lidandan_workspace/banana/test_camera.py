# -*- coding:utf-8 -*-
import cv2

#打开摄像头
cap=cv2.VideoCapture(0)
#设定摄像头采集分辨率为320x230
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

n = 0;
#如果摄像头成功打开
while(cap.isOpened()):
    #读取摄像头采集的当前帧图片数据
    success, frame=cap.read()
    #如果成功读取摄像头
    if (success):
        #显示当前帧图片
        cv2.imshow('testCamera', frame)
        #获取用户按键
        key = cv2.waitKey(1);
        #如果用户按下回车键
        if ( key & 0xFF == 10 ):
            filename = '/home/nvidia/workspace/banana/' + str(n) + '.jpg'
            #按照指定文件名存储当前帧图片
            cv2.imwrite(filename, frame)
            print ('Capture ' + filename)
            n = n + 1
        #如果用户按下取消键
        elif ( key & 0xFF == 27 ): 
            break
    #如果读取摄像头失败
    else:
        break

#释放摄像头
cap.release()

