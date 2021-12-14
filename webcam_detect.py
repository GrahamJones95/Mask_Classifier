import cv2
import label_detect
import sys

filepath = sys.argv[1]
detector = label_detect.detector(filepath)
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
score=0
thicc=2


while(True):
    ret, frame = cap.read()
    height,width = frame.shape[:2]
    label = detector.classify_face(frame)
 
    cv2.putText(frame,"Using "+detector.filepath,(100,height-50), font, 1,(255,255,255),1,cv2.LINE_AA)
    cv2.putText(frame,str(label),(100,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
