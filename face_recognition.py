import cv2 as cv
import os 
import numpy as np
people=[]
features=[]
labels=[]
haar_cascade=cv.CascadeClassifier('/Applications/Python 3.11/haarcascade_frontalface_default.xml')
for i in os.listdir('/Users/praavinkumar/OPEN_CV/Face_Recognition/Celebrity Faces Dataset'):
    people.append(i)
dir='/Users/praavinkumar/OPEN_CV/Face_Recognition/Celebrity Faces Dataset'
def image_function():
    for person in people:
        file_path=os.path.join(dir,person)
        label=people.index(person)
        for image in os.listdir(file_path):
            image_path=os.path.join(file_path,image)
            actual_image=cv.imread(image_path)
            gray=cv.cvtColor(actual_image,cv.COLOR_BGR2GRAY)
            detection=haar_cascade.detectMultiScale(gray,1.1,4)
            for(x,y,w,h) in detection:
                face=gray[y:y+h,x:x+w]
                features.append(face)
                labels.append(label)
image_function()
features=np.array(features,dtype='object')
labels=np.array(labels)
face_recognizer=cv.face.LBPHFaceRecognizer_create()
face_recognizer.train(features,labels)
face_recognizer.save('face_recognizer.yml')
np.save('feautures.npy',features)
np.save('labelss.npy',labels)
