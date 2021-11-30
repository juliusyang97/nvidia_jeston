#-*- coding:utf-8 -*-
import dlib
import cv2
import os
import FaceNetModel

#人脸区域检测器
detector = dlib.get_frontal_face_detector()
#获取FaceNet网络模型
face_net = FaceNetModel.get_FaceNetModel('20180402-114759.pb')

#创建已知人脸列表
def create_known(known_path):
    #人名列表
    person_names = []
    #特征向量列表
    face_features = []

    print("Creating Known Face Library...")
    #遍历指定目录中所有文件
    for file_name in os.listdir(known_path):
        #不是图像文件便不做处理
        if (file_name.find('png')<0) and (file_name.find('jpg')<0):
            continue

        #载入当前图像文件
        image = cv2.imread(os.path.join(known_path,file_name))
        #将摄图像文件从BGR格式转为RGB格式
        rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #第一步：检测出图像中所有人脸区域矩形
        dets = detector(rgb_img)
        #检测不到人脸便不做处理
        if (len(dets) == 0):
            continue

        #取出第一张人脸
        det = dets[0]
        #根据人脸区域矩形对原图进行裁剪
        face_img = rgb_img[det.top():det.bottom(),det.left():det.right()]
        #第二步：基于FaceNet计算人脸的描述特征向量
        descriptor = face_net.get_decriptor(face_img)

        #将文件名(去除扩展名)作为人名添加到人名列表中
        person_name = file_name[:file_name.rfind('.')]
        person_names.append(person_name)
        #将人脸描述特征向量添加到特征向量列表中
        face_features.append(descriptor)
        print('Appending [' + person_name + '] ...')

        #在摄像头采集的原图像上绘制人脸区域矩形
        cv2.rectangle(image,(det.left(), det.top()), 
                            (det.right(), det.bottom()), (0, 255, 0))
        #在摄像头采集的原图像上输出人名
        cv2.putText(image, person_name, (det.left(), det.top()), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        #显示绘制了矩形并输出了人名的图像
        cv2.imshow('image', image)
        #用户在显示图像窗口上按任意键继续
        cv2.waitKey(0)

    print("Known Face Library Created!")
    #返回人名列表和人脸特征向量列表
    return person_names, face_features
