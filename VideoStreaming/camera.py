#import the necessary packages

import cv2

labels_dict = { 0 : 'dog', 1 : 'cat'}
color_dict = {0 : (0, 0, 255), 1 : (0, 255, 0)}

# global loaded_model
# graph1 = Graph()
# with graph1.as_default():
#     session1 = Session(graph = graph1)
#     with session1.as_default():
#         loaded_model = pickle.load(open('combined_model.p', 'rb'))

#defining pet detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalcatface.xml")
ds_factor = 0.6

class VideoCamera(object):
    def __init__(self):
        #capturing video
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        #releasing camera
        self.video.release()
    
    def get_frame(self):
        #extracting frames
        success, image = self.video.read()
        image=cv2.resize(image,None,fx=ds_factor,fy=ds_factor,interpolation=cv2.INTER_AREA)
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        face_rects=face_cascade.detectMultiScale(gray,1.3,5)
        for (x,y,w,h) in face_rects:
        	cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
        	break
        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()
