#Camera Test

import numpy as np
import tensorflow as tf
import cv2
from tensorflow import keras
from tensorflow.keras.models import load_model

#labeling data: 
labels_dict = { 0 : 'dog', 1 : 'cat'}
color_dict = {0 : (0, 0, 255), 1 : (0, 255, 0)}
size = 4
webcam = cv2.VideoCapture(0)

# load xml file
cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalcatface.xml") 
#assuming cat face cascade will show dog faces, too

#load cnn model
cnn = load_model('model')

# while True:
#     ret, frame = webcam.read()
#     img = cv2.flip(frame, 1)
#     face = cascade.detectMultiScale(img, 1.3, 5)
#     for (x, y, w, h) in face:
#         cv2.rectangle(img, (x,y), (x+w, y+h), (0, 128,0), 2)
#     cv2.imshow('image', img)
#     k = cv2.waitKey(1)
#     if k%256 == 27:
#         break

# cam.release()
# cv2.DestroyAllWindows

while True:
    (rval, im) = webcam.read()
    im = cv2.flip(im, 1, 1) #to mirror the image

    #resize for speed
    mini = cv2.resize(im, (im.shape[1] // size, im.shape[0] // size))  
    
    #detect Multiscale
    faces = cascade.detectMultiScale(mini)

    #draw rectangle around cat face
    for f in faces:
        (x,y,w,h) = [v*size for v in f] #scale
        #save the rectangles
        face_img = im[y:y+h, x:x+w]
        resized = cv2.resize(face_img, (160, 160))
        normalized = resized/255.0
        reshaped = np.reshape(normalized, (1,150,150,3))
        reshaped = np.vstack([reshaped])
        result = cnn.predict(reshaped)
        print(result)

        label = argmax(result, axis = 1)[0]

        cv2.rectange(im, (x, y), (x+w, y+h), color_dict[label], 2)
        cv2.rectangle(im, (x, y-40), (x+w, y), color_dict[label], -1)
        cv2.putText(im, labels_dict[label], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    #display the image
    cv2.imshow('LIVE', im)
    key = cv2.waitKey(0)

    #break out of loop on pressing escape key
    if key == 27:
        break

#stop video
webcam.release()

#close all windows
cv2.destroyAllWindows()


