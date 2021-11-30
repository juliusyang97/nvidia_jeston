#-*- coding:utf-8 -*-
import cv2
import numpy as np
import FaceNetKnown
import time

#创建已知人名列表和特征向量列表
person_names, face_features = FaceNetKnown.create_known('known')

#打开摄像头
cap=cv2.VideoCapture(0)
cv2.namedWindow('cam_facenet', cv2.WINDOW_NORMAL)

#如果摄像头成功打开
while(cap.isOpened()):
    #读取摄像头采集的当前帧图片数据
    success, frame=cap.read()
    #如果成功读取摄像头
    if (success):
        rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #第一步：检测出图像中所有人脸区域矩形
        dets = FaceNetKnown.detector(rgb_img)
        #逐个矩形进行处理
        for det in dets:
            #无效矩形不予处理
            if (det.top()<0) or (det.bottom()<0) or (det.left()<0) or (det.right()<0):
                continue;

            #开始计时
            time0 = time.time()
            #根据人脸区域矩形对原图进行裁剪
            face_img = rgb_img[det.top():det.bottom(),det.left():det.right()]
            face_img = rgb_img[det.top():det.bottom(),det.left():det.right()]
            #第二步：基于FaceNet计算人脸的描述特征向量
            descriptor = FaceNetKnown.face_net.get_decriptor(face_img)
            #输出本次推理耗时(单位是秒)
            print("Regnition Cost" , time.time()-time0)

            #最小距离初始值
            min_dist = 1.0
            #人名初始值
            person_name = 'unknown'
            #第三步：逐个已知特征向量遍历查找最接近的匹配
            for i in range(len(face_features)):
                #计算测试图像特征向量和已知特征向量的欧式距离
                dist = np.linalg.norm(np.subtract(descriptor, face_features[i]))
                #找到最小欧式距离对应的人名
                if dist < min_dist:
                    min_dist = dist
                    person_name = person_names[i]
            #在摄像头采集的原图像上绘制人脸区域矩形
            cv2.rectangle(frame,(det.left(), det.top()), 
                                (det.right(), det.bottom()), (0, 255, 0))
            #在摄像头采集的原图像上输出人名
            cv2.putText(frame, person_name, (det.left(), det.top()), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        #显示绘制了矩形并输出了人名的图像
        cv2.imshow('cam_facenet', frame)
        #如果用户按下取消键则退出循环
        if cv2.waitKey(1) == 27:
            break
#释放摄像头
cap.release()
