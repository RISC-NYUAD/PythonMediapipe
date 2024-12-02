import cv2
import time
import mediapipe as mp

mp_holistic = mp.solutions.holistic
holistic_model = mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils


capture = cv2.VideoCapture(0)

if not capture.isOpened():
  print("Failed to open video source")
  exit()

source_fps = int(capture.get(cv2.CAP_PROP_FPS))
source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
source_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
print("Source video FPS: {source_fps}, Dimensions: {source_width}x{source_height}")


fourcc = cv2.VideoWriter_fourcc(*"mp4v")
outVideo = cv2.VideoWriter("video.mp4", fourcc, source_fps, (source_width, source_height))
outResult = cv2.VideoWriter("result.mp4", fourcc, source_fps, (source_width, source_height))

stream_file = open("stream.txt", "w")

previousTime = 0

while capture.isOpened():

    ret, frame = capture.read()

    if not ret or frame is None:
      print("Failed to read frame.")
      break
 

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = holistic_model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
 
    mp_drawing.draw_landmarks(
      image, 
      results.right_hand_landmarks, 
      mp_holistic.HAND_CONNECTIONS
    )
 
    mp_drawing.draw_landmarks(
      image, 
      results.left_hand_landmarks, 
      mp_holistic.HAND_CONNECTIONS
    )

    mp_drawing.draw_landmarks(
      image,
      results.pose_landmarks,
      mp_holistic.POSE_CONNECTIONS
    )

    if results.pose_landmarks:
      stream_file.write("Pose Landmarks:\n")
      for i, landmark in enumerate(results.pose_landmarks.landmark):
        line = f"Landmark {i}: x={landmark.x:.4f}, y={landmark.y:.4f}, z={landmark.z:.4f}, visibility={landmark.visibility:.4f}\n"
        stream_file.write(line)
        print(line.strip())

    if results.right_hand_landmarks:
      stream_file.write("Right Hand Landmarks:\n")
      for i, landmark in enumerate(results.right_hand_landmarks.landmark):
          line = f"Landmark {i}: x={landmark.x:.4f}, y={landmark.y:.4f}, z={landmark.z:.4f}\n"
          stream_file.write(line)
          print(line.strip())

    if results.left_hand_landmarks:
      stream_file.write("Left Hand Landmarks:\n")
      for i, landmark in enumerate(results.left_hand_landmarks.landmark):
          line = f"Landmark {i}: x={landmark.x:.4f}, y={landmark.y:.4f}, z={landmark.z:.4f}\n"
          stream_file.write(line)
          print(line.strip())


    currentTime = time.time()
    fps = 1 / (currentTime-previousTime)
    previousTime = currentTime
     

    cv2.putText(image, str(int(fps))+" FPS", (10, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
    cv2.imshow("MediaPipe Holistic", image)

    outVideo.write(frame)
    outResult.write(image)
 
    if cv2.waitKey(5) & 0xFF == ord("q"):
        break
 
capture.release()
outVideo.release()
outResult.release()
stream_file.close()
cv2.destroyAllWindows()

