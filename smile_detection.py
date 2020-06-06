# Smile Detection using Face Landmarks
# LinkedIn: https://www.linkedin.com/in/tanmoymunshi/
# GitHub: https://github.com/tanmoy1999


import cv2
import numpy as np
import requests
import dlib

url = 'http://192.168.0.2:8080/shot.jpg'


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

while True:
    img_resp = requests.get(url)
    img_arr = np.array(bytearray(img_resp.content), dtype = np.uint8)
    img = cv2.imdecode(img_arr,-1)
    img = cv2.resize(img,(1080,720))

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    faces = detector(gray)

    for face in faces:
        x1 = face.left()
        y1 = face.top()
        y2 = face.bottom()
        x2 = face.right()

        landmarks = predictor(gray,face)

        '''
        for n in range(0,68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y

            print((landmarks.part(48).x,landmarks.part(48).y))
            cv2.circle(img,(x,y),4,(255,0,0),-1)
        '''
        x1 = landmarks.part(48).x
        y1 = landmarks.part(48).y

        x2 = landmarks.part(54).x
        y2 = landmarks.part(54).y

        print((x1,y1),(x2,y2))
        
        len_line = x2-x1
        print(len_line)
        if len_line > 105:
            cv2.putText(img,"Happy Face",(20,20),cv2.FONT_HERSHEY_COMPLEX,.7,(0, 255, 0), 2)
        else:
            cv2.putText(img,"Normal Face",(20,20),cv2.FONT_HERSHEY_COMPLEX,.7,(0, 255, 0), 2)
              
        line = cv2.line(img,(x1,y1),(x2,y2),(255,0,0),4)

        cv2.circle(img,(x1,y1),4,(255,0,0),-1)
        cv2.circle(img,(x2,y2),4,(255,0,0),-1)

        


    cv2.imshow('frame',img)
        
    key = cv2.waitKey(1)
    if key == 27:
        break


