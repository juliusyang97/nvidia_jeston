# -*- coding: utf-8 -*-
import cv2
import numpy as np
import FaceNetKnown

#创建已知人名列表和特征向量列表
person_names, face_features = FaceNetKnown.create_known('known')

#载入测试图像文件
image = cv2.imread('nina_victory.jpg')
rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#检测出图像中所有人脸区域矩形
dets = FaceNetKnown.detector(rgb_img)

#逐个矩形进行处理
for det in dets:
    face_img = rgb_img[det.top():det.bottom(),det.left():det.right()]
    #根据标志点计算人脸的描述特征向量
    descriptor = FaceNetKnown.face_net.get_decriptor(face_img)

    #最小距离初始值
    min_dist = 1.0
    #人名初始值
    person_name = 'unknown'
    #逐个已知特征向量遍历
    for i in range(len(face_features)):
        #计算测试图像特征向量和已知特征向量的欧式距离
        dist = np.linalg.norm(np.subtract(descriptor, face_features[i]))
        print(person_names[i], dist)
        #找到最小欧式距离对应的人名
        if dist < min_dist:
            min_dist = dist
            person_name = person_names[i]
    print('Found [' + person_name + '] !')

    #在摄像头采集的原图像上绘制人脸区域矩形
    cv2.rectangle(image,(det.left(), det.top()), 
                        (det.right(), det.bottom()), (0, 255, 0))
    #在摄像头采集的原图像上输出人名
    cv2.putText(image, person_name, (det.left(), det.top()), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    #显示绘制了矩形并输出了人名的图像
    cv2.imshow('image', image)
    #等待用户按下任意键退出
    cv2.waitKey(0)
