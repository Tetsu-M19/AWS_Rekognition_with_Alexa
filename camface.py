 # -*- coding: utf-8 -*-
import cv2 as cv
import time
import boto3
import subprocess
import datetime

BUCKET_NAME = 'XXXXXXXXXX'
SOURCE_IMAGE = ['XXXXXXXXXX.jpg', 'XXXXXXXXXX.jpg', 'XXXXXXXXXX.jpg']

cap = cv.VideoCapture(-1)

# image_name
fn = 'detect_face.jpg'

# 顔検出のための学習元データを読み込む
face_cascade = cv.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
'''
#WebCamera_Setting
WIDTH = 640
HEIGHT = 480
FPS = 30

#decode
def decode_fourcc(v):
    v = int(v)
    return "".join([chr((v >> 8 * i) & 0xFF) for i in range(4)])
    
# フォーマット・解像度・FPSの設定
cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc('M','J','P','G'))
cap.set(cv.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, HEIGHT)
cap.set(cv.CAP_PROP_FPS, FPS)

# フォーマット・解像度・FPSの取得
fourcc = decode_fourcc(cap.get(cv.CAP_PROP_FOURCC))
width = cap.get(cv.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv.CAP_PROP_FRAME_HEIGHT)
fps = cap.get(cv.CAP_PROP_FPS)
print("fourcc:{} fps:{}　width:{}　height:{}".format(fourcc, fps, width, height))
'''
def compare_faces(targetFile):
    count = 0
    detect_name = []
    client=boto3.client('rekognition')
    
    for i in range(len(SOURCE_IMAGE)):
        imageTarget=open(targetFile,'rb')
        response=client.compare_faces(SimilarityThreshold=80,
                                      SourceImage={
                                            'S3Object': {
                                                'Bucket': BUCKET_NAME,
                                                'Name': SOURCE_IMAGE[i],
                                            }
                                        },
                                      TargetImage={'Bytes': imageTarget.read()})

        for faceMatch in response['FaceMatches']:
            position = faceMatch['Face']['BoundingBox']
            similarity = str(faceMatch['Similarity'])
            print('The face at matches with' + ' ' + SOURCE_IMAGE[i] + ' ' + similarity + '% similarity')
            
            if SOURCE_IMAGE[i] == "XXXXXXXXXX.jpg":
                detect_name.append("XXXXXXXXXX")
            
        count += len(response['FaceMatches']) 

    imageTarget.close()
    print(detect_name)
    return count, detect_name

while True:
    # 1フレームずつ取得する。
    ret, frame = cap.read()
    frame_detect = frame.copy()
    
    #フレームが取得できなかった場合は、画面を閉じる
    if not ret:
        break
    
    # 顔検出の処理効率化のために、写真の情報量を落とす（モノクロにする）
    grayimg = cv.cvtColor(frame_detect, cv.COLOR_BGR2GRAY)
    
    # 顔検出を行う
    facerect = face_cascade.detectMultiScale(grayimg, scaleFactor=1.2,
                                             minNeighbors=2, minSize=(100, 100))

    Key=cv.waitKey(10)
    # 顔が検出された場合
    if len(facerect) > 0:
        # 検出した場所すべてに赤色で枠を描画する
        for rect in facerect:
            cv.rectangle(frame_detect, tuple(rect[0:2]), tuple(rect[0:2]+rect[2:4]), (0, 0, 255), thickness=3)
            
        # そのときの画像を保存する
        if Key==ord("c"):
            subprocess.run(["./alexa_remote_control.sh", "-e", "sound:bell_02"])
            cv.imwrite(fn, frame)
            print("capture!!")
            
            target_file='/home/pi/detect_face.jpg'
            face_count, face_name = compare_faces(target_file)
            print("Face matches: " + str(face_count))
            
            if face_count == 1:
                print("Unlock!!")
                dt_now = datetime.datetime.now()
                subprocess.run(["./alexa_remote_control.sh", "-e", "speak:こんにちは, {} さん. 今日は, {} です.".format(face_name, dt_now.strftime('%Y年%m月%d日'))])
            elif face_count > 1:
                print("Error")
                subprocess.run(["./alexa_remote_control.sh", "-e", "speak:一度に認証できるのはひとりまでです. もう一度やりなおしてください."])
            else:
                print("Error")
                subprocess.run(["./alexa_remote_control.sh", "-e", "speak:認証できませんでした. もう一度やりなおしてください."])
            time.sleep(1)

    # カメラから取得した映像を表示する
    cv.imshow('camera', frame_detect)

    # esc_key
    if Key==27:
        break

# 表示したウィンドウを閉じる
cap.release()
cv.destroyAllWindows()
