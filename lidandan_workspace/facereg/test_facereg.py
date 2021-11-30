#-*- coding:utf-8 -*-
import dlib
import numpy as np
import FaceKnown

#创建已知人名列表和特征向量列表
person_names, face_features = FaceKnown.create_known('known')
#创建显示窗口
win = dlib.image_window()
#载入测试图像文件
img = dlib.load_rgb_image('Girls.png')
#第一步：检测出图像中所有人脸区域矩形
dets = FaceKnown.detector(img)

#逐个矩形进行处理
for det in dets:
    #第二步：检测出当前人脸区域内的标志点
    shape = FaceKnown.predictor(img, det)
    #第三步：结合标志点计算人脸的描述特征向量
    descriptor = FaceKnown.recognizer.compute_face_descriptor(img, shape)

    #最小距离初始值
    min_dist = 1.0
    #人名初始值
    person_name = 'unknown'
    #第四步：逐个已知特征向量遍历查找最接近的匹配
    for i in range(len(face_features)):
        #计算测试图像特征向量和已知特征向量的欧式距离
        dist = np.linalg.norm(np.subtract(descriptor, face_features[i]))
        print(person_names[i], dist)
        #找到最小欧式距离对应的人名
        if dist < min_dist:
            min_dist = dist
            person_name = person_names[i]
    print('Found [' + person_name + '] !')

    #窗口清空后显示当前图像
    win.clear_overlay()
    win.set_image(img)
    #在原图像上叠加显示标志点
    win.add_overlay(shape, dlib.rgb_pixel(0,255,255))
    #在原图像上叠加显示各人脸区域矩形
    win.add_overlay(det,dlib.rgb_pixel(255,255,0))

    #在Terminal窗口点击任意键继续
    dlib.hit_enter_to_continue()
