import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

def DrawLandmark(index, image):
  RADIUS = 3
  coords = results.multi_face_landmarks[0].landmark[index]
  height, width = image.shape[:2]
  coordX = int(coords.x*width)
  coordY = int(coords.y*height)
  cv2.circle(image, (coordX, coordY), RADIUS, (0, 255, 0), -RADIUS)
  return (coordX, coordY)

def Ifface(p1, p2, p3):
  d1 = (np.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2))
  d2 = (np.sqrt((p1[0]-p3[0])**2+(p1[1]-p3[1])**2))
  thresh = np.abs(d2-d1)
  if thresh>10:
    return False
  else:
    return True
  


# For webcam input:
cap = cv2.VideoCapture(0)
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
        print(Ifface(middle,rightEye,leftEye))      
        
    cv2.imshow('MediaPipe FaceMesh', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()