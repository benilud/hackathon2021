import cv2
import mediapipe as mp
import numpy as np
import sys


if len(sys.argv) == 1 :
  CAMERA = 0
else :
  CAMERA = int(sys.argv[1])


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

def DrawLandmark(index, image):
  RADIUS = 3
  coords = results.multi_face_landmarks[0].landmark[index]
  height, width = image.shape[:2]
  coordX = int(coords.x*width)
  coordY = int(coords.y*height)
  #cv2.circle(image, (coordX, coordY), RADIUS, (0, 255, 0), -RADIUS)
  return (coordX, coordY)

def linePerp(coord3, pente, image):
  if pente == 0:
    pente = 0.001

  perpCoefDir = -1/pente
  x4 = x3+1
  y4 = y3+perpCoefDir
  coord4 = (x4,y4)
  return linePoints(coord3, coord4, image)


def linePoints(coord1,coord2,image):
  Point1 = (0,0)
  Point2 = (0,0)
  height, width = image.shape[:2]
  if coord1[0] == coord2[0]:
    a = 0
    b = 1

    Point1 = (0, coord1[0])
    Point2 = (height, coord1[1])
  else:
    a = (coord2[1]-coord1[1])/(coord2[0]-coord1[0])
    
    if a==0:
      b = coord1[1]
      Point1 = (0, coord1[1])
      Point2 = (width, coord1[1])
    
    else :
      b = (coord1[1]*coord2[0]-coord2[1]*coord1[0])/(coord2[0]-coord1[0])

      if (coord1[0]>0 and coord1[0]<width):
        Point1 = (-b/a,0);
        Point2 = ((height-b)/a,height)

  Point1 = (int(Point1[0]),int(Point1[1]))
  Point2 = (int(Point2[0]),int(Point2[1]))
  return(Point1,Point2,a,b)

def parallel(point1, point2, offset, nb_line, image):
  alpha = np.arctan2(point2[0]-point1[0],point2[1]-point1[1])
  diag = np.sqrt(width**2+height**2)/100
  offsetX = offset*np.cos(alpha)*diag
  offsetY = -offset*np.sin(alpha)*diag
  for i in range(1, nb_line+1):
    P1 = linePoints((point1[0]+offsetX*i, point1[1]+offsetY*i), (point2[0]+offsetX*i, point2[1]+offsetY*i), image)[0]
    P2 = linePoints((point1[0]+offsetX*i, point1[1]+offsetY*i), (point2[0]+offsetX*i, point2[1]+offsetY*i), image)[1]
    
    cv2.line(image,P1,P2,(255,0,0),1)

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
  TEXT = 'Merci de mettre le patient de face'

  height, width = image.shape[:2]
  textsize = cv2.getTextSize(TEXT, FONT, 1, 2)[0]
  textX = int((width - textsize[0]) / 2)
  textY = int((height - textsize[1]) / 2)

  box_coords = ((textX - OFFSET_BGRD, textY + OFFSET_BGRD), (textX + textsize[0] + OFFSET_BGRD, textY - textsize[1] - OFFSET_BGRD))
  cv2.rectangle(image, box_coords[0], box_coords[1], RECTANGLE_BGR, cv2.FILLED)
  image = cv2.putText(image, TEXT, (textX,textY), FONT, 1, (0,0,255), 2)


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

    height, width = image.shape[:2]

    pente = 0    

    if results.multi_face_landmarks:
      for face_landmarks in results.multi_face_landmarks:
        leftEye = DrawLandmark(359, image)
        rightEye = DrawLandmark(130, image)
        middle = DrawLandmark(168, image)

      if (Ifface(middle,rightEye,leftEye)==False):
        Displayalert(image)
      else:
        
        lineP = linePoints(leftEye,rightEye,image)
        cv2.line(image,lineP[0],lineP[1],(0,255,0),1)
        parallel(leftEye,rightEye,3,10,image)
        pente = lineP[2]  
      

        
        coords = results.multi_face_landmarks[0].landmark[168]
        
        x3 = int(coords.x*width)
        y3 = int(coords.y*height)
        coord3 = (x3,y3)

        linePerpCoords = linePerp(coord3, pente, image)
        cv2.line(image,linePerpCoords[0],linePerpCoords[1],(0,255,0),1)
        
    cv2.imshow('Face ILITY viewer', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()