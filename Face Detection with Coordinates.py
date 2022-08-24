#Importing modules 
import numpy as np
import cv2 

#Defining what video source to use: 0 for webcam, "file name.mp4" for 
#video saved in same folder as this code.
video_path=0
video = cv2.VideoCapture(0)
window_name = f"Detected Objects in webcam feed"

#Uses last 200 frames to remove background for easier detection.
back_sub = cv2.createBackgroundSubtractorMOG2(history=200, 
        varThreshold=10, detectShadows=True)

#Create kernel for morphological operation
kernel = np.ones((20,20),np.uint8)



while True: 
    
    # Ret is a boolean that returns true if the frame is available. 
    # Frame is an array that contains the information of the image from the webcam. Also contains the fps
    # of the video feed. 
    ret,frame = video.read()

    #Breaks if no video detected. 

    if not ret:
        break

    #Opens a resizale window for the webcam image to live in. 
    cv2.namedWindow(window_name,cv2.WINDOW_NORMAL)

    #Convert webcam image to grey for easier object detection. 
    grey_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    #Defining what object classifier to use. 
    cascade_classifier = cv2.CascadeClassifier(f"{cv2.data.haarcascades}haarcascade_frontalface_alt.xml") #testing with face classifier.

    #Detect objects
    detected_objects = cascade_classifier.detectMultiScale(grey_image, minSize=(50,50))

    #Highlighting detected objects. 
    if len(detected_objects) != 0:
        for (x,y,h,w) in detected_objects:
            
            #Green rectangle around detected object. 
            cv2.rectangle(frame, (x,y),((x+h), (y+w)),(0,255,0),5) #green rectangle of thickness 5.
           
           #Gives centre position of bounding rectangle
            x2 = x + int(h/2)
            y2 = y + int(w/2)
           
           #Blue circle on centre of bounding rectangle 
            cv2.circle(frame,(x2,y2),5,(255,0,0),5)
           
           #Centre of bounding box coordinates text. 
            text = "x: " + str(x2) + ", y: " + str(y2)
        
           #Prints coords of centre to console.         
            print('Centre of contour box:',text)


        #Add x and y coords to image frame. 
            cv2.putText(frame, text, (5, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 5)
        

    #Display image with detections
    cv2.imshow(window_name,frame)
    

    if cv2.waitKey(1) == 27:
        break

#Close all windows and turn off webcam.
video.release()
cv2.destroyAllWindows()


