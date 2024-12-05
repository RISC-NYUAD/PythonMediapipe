import cv2
import numpy as np
import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time 

VisionRunningMode = mp.tasks.vision.RunningMode
pose_options = vision.PoseLandmarkerOptions(
    base_options=python.BaseOptions(
        model_asset_path="models/pose_landmarker_full.task",
        delegate=python.BaseOptions.Delegate.CPU
    ),
    running_mode=VisionRunningMode.VIDEO,
    output_segmentation_masks=True,

    min_pose_detection_confidence = 0.3,
    min_pose_presence_confidence = 0.3,
    min_tracking_confidence = 0.3,
)
pose_detector = vision.PoseLandmarker.create_from_options(pose_options)

hand_options = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(
        model_asset_path="models/hand_landmarker.task",
        delegate=python.BaseOptions.Delegate.CPU
    ),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence = 0.3,
    min_hand_presence_confidence = 0.3,
    min_tracking_confidence = 0.3,
)
hand_detector = vision.HandLandmarker.create_from_options(hand_options)

def draw_pose_landmarks(rgb_image, detection_result):
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_frame = np.copy(rgb_image)

    for pose_landmarks in pose_landmarks_list:
        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend(
            [
                landmark_pb2.NormalizedLandmark(
                    x=landmark.x, y=landmark.y, z=landmark.z
                )
                for landmark in pose_landmarks
            ]
        )
        solutions.drawing_utils.draw_landmarks(
            annotated_frame,
            pose_landmarks_proto,
            solutions.pose.POSE_CONNECTIONS,
            solutions.drawing_utils.DrawingSpec(
                color=(255, 0, 0), thickness=2, circle_radius=2
            ),
            solutions.drawing_utils.DrawingSpec(
                color=(0, 255, 0), thickness=2, circle_radius=2
            ),
        )
    return annotated_frame

def draw_hand_landmarks(rgb_image, detection_result):
    hand_landmarks_list = detection_result.hand_landmarks
    annotated_frame = np.copy(rgb_image)

    for hand_landmarks in hand_landmarks_list:
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend(
            [
                landmark_pb2.NormalizedLandmark(
                    x=landmark.x, y=landmark.y, z=landmark.z
                )
                for landmark in hand_landmarks
            ]
        )
        solutions.drawing_utils.draw_landmarks(
            annotated_frame,
            hand_landmarks_proto,
            solutions.hands.HAND_CONNECTIONS,
            solutions.drawing_utils.DrawingSpec(
                color=(255, 0, 0), thickness=2, circle_radius=2
            ),
            solutions.drawing_utils.DrawingSpec(
                color=(0, 255, 0), thickness=2, circle_radius=2
            ),
        )
    return annotated_frame

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

while capture.isOpened():
    ret, frame = capture.read()

    if not ret or frame is None:
        print("Failed to read frame.")
        break

    frame.flags.writeable = False
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

    timestamp = int(time.time() * 1000)
    pose_result = pose_detector.detect_for_video(mp_frame, timestamp)
    hand_result = hand_detector.detect_for_video(mp_frame, timestamp)

    frame.flags.writeable = True
    annotated_frame = draw_pose_landmarks(frame, pose_result)
    annotated_frame = draw_hand_landmarks(annotated_frame, hand_result)

    currentTime = time.time()
    fps = 1 / (currentTime - previousTime)
    previousTime = currentTime

    cv2.putText(annotated_frame, f"{int(fps)} FPS", (10, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
    annotated_bgr_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
    cv2.imshow("Pose and Hand Detection", annotated_bgr_frame)

    outResult.write(annotated_bgr_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

capture.release()
outResult.release()
cv2.destroyAllWindows()
