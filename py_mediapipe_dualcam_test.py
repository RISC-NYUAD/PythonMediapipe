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

    min_pose_detection_confidence=0.3,
    min_pose_presence_confidence=0.3,
    min_tracking_confidence=0.3,
)
pose_detector = vision.PoseLandmarker.create_from_options(pose_options)

hand_options = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(
        model_asset_path="models/hand_landmarker.task",
        delegate=python.BaseOptions.Delegate.CPU
    ),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.3,
    min_hand_presence_confidence=0.3,
    min_tracking_confidence=0.3,
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

capture0 = cv2.VideoCapture(0)
capture1 = cv2.VideoCapture(1)

if not capture0.isOpened() and not capture1.isOpened():
    print("Failed to open video sources")
    exit()

capture0.set(cv2.CAP_PROP_BUFFERSIZE, 1)
capture0.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
capture0.set(cv2.CAP_PROP_FPS, 30)
capture0.set(3, 1280)
capture0.set(4, 720)

capture1.set(cv2.CAP_PROP_BUFFERSIZE, 1)
capture1.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
capture1.set(cv2.CAP_PROP_FPS, 30)
capture1.set(3, 1280)
capture1.set(4, 720)

source0_fps = int(capture0.get(cv2.CAP_PROP_FPS))
source0_width = int(capture0.get(cv2.CAP_PROP_FRAME_WIDTH))
source0_height = int(capture0.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Source video FPS: {source0_fps}, Dimensions: {source0_width}x{source0_height}")

source1_fps = int(capture1.get(cv2.CAP_PROP_FPS))
source1_width = int(capture1.get(cv2.CAP_PROP_FRAME_WIDTH))
source1_height = int(capture1.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Source video FPS: {source1_fps}, Dimensions: {source1_width}x{source1_height}")

fourcc0 = cv2.VideoWriter_fourcc(*"mp4v")
outResult0 = cv2.VideoWriter("result.mp4", fourcc0, source0_fps, (source0_width, source0_height))

fourcc1 = cv2.VideoWriter_fourcc(*"mp4v")
outResult1 = cv2.VideoWriter("result.mp4", fourcc1, source1_fps, (source1_width, source1_height))

previousTime = 0

while capture0.isOpened() and capture1.isOpened():
    ret0, frame0 = capture0.read()
    ret1, frame1 = capture1.read()

    if not ret0 or frame0 is None:
        print("Failed to read frame from source 0.")
        break

    if not ret1 or frame1 is None:
        print("Failed to read frame from source 1.")
        break

    frame0.flags.writeable = False
    frame0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2RGB)
    mp_frame0 = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame0)

    frame1.flags.writeable = False
    frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
    mp_frame1 = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame1)

    timestamp = int(time.time() * 1000)
    pose_result0 = pose_detector.detect_for_video(mp_frame0, timestamp)
    hand_result0 = hand_detector.detect_for_video(mp_frame0, timestamp)

    timestamp = int(time.time() * 1000)
    pose_result1 = pose_detector.detect_for_video(mp_frame1, timestamp)
    hand_result1 = hand_detector.detect_for_video(mp_frame1, timestamp)

    frame0.flags.writeable = True
    frame1.flags.writeable = True


    annotated_frame0 = draw_pose_landmarks(frame0, pose_result0)
    annotated_frame0 = draw_hand_landmarks(annotated_frame0, hand_result0)

    annotated_frame1 = draw_pose_landmarks(frame1, pose_result1)
    annotated_frame1 = draw_hand_landmarks(annotated_frame1, hand_result1)

    currentTime = time.time()
    fps = 1 / (currentTime - previousTime)
    previousTime = currentTime

    cv2.putText(annotated_frame0, f"{int(fps)} FPS", (10, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
    annotated_bgr_frame0 = cv2.cvtColor(annotated_frame0, cv2.COLOR_RGB2BGR)

    cv2.putText(annotated_frame1, f"{int(fps)} FPS", (10, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
    annotated_bgr_frame1 = cv2.cvtColor(annotated_frame1, cv2.COLOR_RGB2BGR)



    cv2.imshow("Source 0", annotated_bgr_frame0)
    cv2.imshow("Source 1", annotated_bgr_frame1)

    outResult0.write(annotated_bgr_frame0)
    outResult1.write(annotated_bgr_frame1)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

capture0.release()
capture1.release()
outResult0.release()
outResult1.release()
cv2.destroyAllWindows()
