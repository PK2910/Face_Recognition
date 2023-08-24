import cv2 as cv
import numpy as np
from face_recognition import people
haar_cascade=cv.CascadeClassifier('/Applications/Python 3.11/haarcascade_frontalface_default.xml')
np.load('feautures.npy',allow_pickle=True)
np.load('labelss.npy')
face_recognizer=cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_recognizer.yml')
image=cv.imread('Face_Recognition/Celebrity Faces Dataset/Leonardo DiCaprio/006_30010640.jpg')
gray=cv.cvtColor(image,cv.COLOR_BGR2GRAY)
cv.imshow('gray',gray)
#Detect the face in the image
face=haar_cascade.detectMultiScale(gray,1.1,4)
for(x,y,w,h) in face:
    face_cropped=gray[y:y+h,x:x+w]
    label,confidence=face_recognizer.predict(face_cropped)
    print(f"Label is {people[label]} with confidence of {confidence}")
    cv.putText(image,str(people[label]),(20,20),cv.FONT_HERSHEY_COMPLEX,1.0,(0,255,0),2)
    cv.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
cv.imshow('Celebrity',image)
cv.waitKey(0)

