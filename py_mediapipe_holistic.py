import cv2
import numpy as np
import mediapipe as mp
import time 
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

capture = cv2.VideoCapture(0)
#capture = cv2.VideoCapture('test.mp4')

if not capture.isOpened():
    print("Failed to open video source")
    exit()

capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
capture.set(cv2.CAP_PROP_FPS, 30)
capture.set(3, 1920)
capture.set(4, 1080)


source_fps = int(capture.get(cv2.CAP_PROP_FPS))
source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
source_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Source video FPS: {source_fps}, Dimensions: {source_width}x{source_height}")

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
outResult = cv2.VideoWriter("result.mp4", fourcc, source_fps, (source_width, source_height))

previousTime = 0

with mp_holistic.Holistic(
  static_image_mode=False,
  model_complexity=0,
  #model_complexity=1,
  #model_complexity=2,
  enable_segmentation=True,
  min_detection_confidence=0.5,
  min_tracking_confidence=0.5) as holistic:
  while capture.isOpened():
    
    ret, frame = capture.read()
    if not ret:
      print("Failed to read frame.")
      break

    frame.flags.writeable = False
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(frame)

    frame.flags.writeable = True
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    mp_drawing.draw_landmarks(
        frame,
        results.pose_landmarks,
        mp_holistic.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
        connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0),thickness=2,circle_radius=2))
    mp_drawing.draw_landmarks(
        frame,
        results.left_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS,
        landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
        connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0),thickness=2,circle_radius=2))
    mp_drawing.draw_landmarks(
        frame,
        results.right_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS,
        landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
        connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0),thickness=2,circle_radius=2))

    currentTime = time.time()
    fps = 1 / (currentTime - previousTime)
    previousTime = currentTime

    cv2.putText(frame, f"{int(fps)} FPS", (10, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

    outResult.write(frame)

    cv2.imshow('MediaPipe Holistic', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

outResult.release()
capture.release()