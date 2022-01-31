import cv2


img_file = 'cars.jpeg'
cap= cv2.VideoCapture('video.mp4')


#pretrained car clssifier
classifier_file = 'cars.xml'

car_tracker = cv2.CascadeClassifier(classifier_file)

while True:
    _, frame = cap.read()  #reading video in frame

    greyscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)   #converting into grey so that face detection is easy

    cars = car_tracker.detectMultiScale(greyscale)

    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 2)


    cv2.imshow('frame', frame)
    if cv2.waitKey(10) == ord('q'):
        break


