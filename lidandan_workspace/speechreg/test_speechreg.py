#coding:utf-8
import speech_recognition
#加载语音识别器
recognizer = speech_recognition.Recognizer()

#从音频文件中获取音频数据
audio_file = speech_recognition.AudioFile('/home/nvidia/Downloads/harvard.wav')
with audio_file as source:
    audio = recognizer.record(source, offset=4, duration=3)
print("Analyzing audio_file...")
#使用Sphinx引擎完成语音识别
print(recognizer.recognize_sphinx(audio))
