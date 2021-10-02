import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

CAMERA = 1

def DrawLandmark(index, image):
  RADIUS = 3
  coords = results.multi_face_landmarks[0].landmark[index]
  height, width = image.shape[:2]
  coordX = int(coords.x*width)
  coordY = int(coords.y*height)
  cv2.circle(image, (coordX, coordY), RADIUS, (0, 255, 0), -RADIUS)
  return (coordX, coordY)

def Ifface(pmiddle, pright, pleft): #test face = front with 3 points p(x,y)
  LIM_TRESHOLD = 25 #fix sensibility of the detection

  def dist(pmiddle,pright):
    return (np.sqrt((pmiddle[0]-pright[0])**2+(pmiddle[1]-pright[1])**2))

  d1 = (dist(pmiddle,pright)/dist(pright,pleft))*100
  d2 =(dist(pmiddle,pleft)/dist(pright,pleft))*100
  thresh = np.abs(d2-d1)
  
  if thresh > LIM_TRESHOLD:
    return False
  else:
    return True

def Displayalert(image):
  RECTANGLE_BGR = (255, 255, 255)
  OFFSET_BGRD = 10
  FONT = cv2.FONT_HERSHEY_SIMPLEX
  TEXT = 'Merci de vous mettre de face'

  height, width = image.shape[:2]
  textsize = cv2.getTextSize(TEXT, FONT, 1, 2)[0]
  textX = int((width - textsize[0]) / 2)
  textY = int((height - textsize[1]) / 2)

  box_coords = ((textX - OFFSET_BGRD, textY + OFFSET_BGRD), (textX + textsize[0] + OFFSET_BGRD, textY - textsize[1] - OFFSET_BGRD))
  cv2.rectangle(image, box_coords[0], box_coords[1], RECTANGLE_BGR, cv2.FILLED)
  image = cv2.putText(image, TEXT, (textX,textY), FONT, 1, (0,0,255), 2)
  cv2.imshow('MediaPipe FaceMesh', image)  


# For webcam input:
cap = cv2.VideoCapture(CAMERA)
with mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # Flip the image horizontally for a later selfie-view display, and convert
    # the BGR image to RGB.
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    results = face_mesh.process(image)

    # Draw the face mesh annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)    

    if results.multi_face_landmarks:
      for face_landmarks in results.multi_face_landmarks:
        leftEye = DrawLandmark(359, image)
        rightEye = DrawLandmark(130, image)
        middle = DrawLandmark(168, image)
        if (Ifface(middle,rightEye,leftEye)==False):
          Displayalert(image)
        
    cv2.imshow('MediaPipe FaceMesh', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()