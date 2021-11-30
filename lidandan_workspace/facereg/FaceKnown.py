#-*- coding:utf-8 -*-
import dlib
import os

#人脸区域检测器
detector = dlib.get_frontal_face_detector()
#人脸标志点检测器
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
#人脸识别模型
recognizer = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')

#创建已知人脸列表
def create_known(known_path):
    #人名列表
    person_names = []
    #特征向量列表
    face_features = []
    #创建显示窗口
    win = dlib.image_window()
    print("Creating Known Face Library...")

    #遍历指定目录中所有文件
    for file_name in os.listdir(known_path):
        #不是图像文件便不做处理
        if (file_name.find('png')<0) and (file_name.find('jpg')<0):
            continue

        #载入当前图像文件
        img = dlib.load_rgb_image(os.path.join(known_path,file_name))
        #第一步：检测出图像中所有人脸区域矩形
        dets = detector(img)
        #检测不到人脸便不做处理
        if (len(dets) == 0):
            continue

        #取出第一张人脸
        det = dets[0]
        #第二步：检测出当前人脸区域内的标志点
        shape = predictor(img, det)
        #第三步：结合标志点计算人脸的描述特征向量
        descriptor = recognizer.compute_face_descriptor(img, shape)

        #将文件名(去除扩展名)作为人名添加到人名列表中
        person_name = file_name[:file_name.rfind('.')]
        person_names.append(person_name)
        #将人脸描述特征向量添加到特征向量列表中
        face_features.append(descriptor)
        print('Appending [' + person_name + '] ...')
        dlib.save_face_chip(img, shape, 'output/'+person_name, size=160, padding=0.25)

        #窗口清空后显示当前图像
        win.clear_overlay()
        win.set_image(img)
        #在原图像上叠加显示标志点
        win.add_overlay(shape, dlib.rgb_pixel(0,255,255))
        #在原图像上叠加显示各人脸区域矩形
        win.add_overlay(det, dlib.rgb_pixel(255,255,0))

        #在Terminal窗口点击任意键继续
        dlib.hit_enter_to_continue()

    print("Known Face Library Created!")
    #返回已知人名列表和已知人脸特征向量列表
    return person_names, face_features
