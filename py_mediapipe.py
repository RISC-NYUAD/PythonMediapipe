import cv2
import numpy as np
import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time

# Create PoseLandmarker object
pose_options = vision.PoseLandmarkerOptions(
    base_options=python.BaseOptions(
        model_asset_path="pose_landmarker.task",
        delegate=python.BaseOptions.Delegate.GPU
    ),
    output_segmentation_masks=True
)
pose_detector = vision.PoseLandmarker.create_from_options(pose_options)

# Create HandLandmarker object
hand_options = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(
        model_asset_path="hand_landmarker.task",
        delegate=python.BaseOptions.Delegate.GPU
    ),
    num_hands=2  # Max number of hands to detect
)
hand_detector = vision.HandLandmarker.create_from_options(hand_options)

# Function to draw pose landmarks
def draw_pose_landmarks(rgb_image, detection_result):
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)

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
            annotated_image,
            pose_landmarks_proto,
            solutions.pose.POSE_CONNECTIONS,
            solutions.drawing_utils.DrawingSpec(
                color=(255, 0, 0), thickness=1, circle_radius=1
            ),
            solutions.drawing_utils.DrawingSpec(
                color=(0, 255, 0), thickness=1, circle_radius=1
            ),
        )
    return annotated_image

# Function to draw hand landmarks
def draw_hand_landmarks(rgb_image, detection_result):
    hand_landmarks_list = detection_result.hand_landmarks
    annotated_image = np.copy(rgb_image)

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
            annotated_image,
            hand_landmarks_proto,
            solutions.hands.HAND_CONNECTIONS,
            solutions.drawing_utils.DrawingSpec(
                color=(255, 0, 0), thickness=1, circle_radius=1
            ),
            solutions.drawing_utils.DrawingSpec(
                color=(0, 255, 0), thickness=1, circle_radius=1
            ),
        )
    return annotated_image

# Initialize webcam capture
capture = cv2.VideoCapture(0)

if not capture.isOpened():
    print("Failed to open video source")
    exit()

source_fps = int(capture.get(cv2.CAP_PROP_FPS))
source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
source_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Source video FPS: {source_fps}, Dimensions: {source_width}x{source_height}")

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
outVideo = cv2.VideoWriter("video.mp4", fourcc, source_fps, (source_width, source_height))
outResult = cv2.VideoWriter("result.mp4", fourcc, source_fps, (source_width, source_height))

previousTime = 0

# Process video feed
while capture.isOpened():
    ret, frame = capture.read()

    if not ret or frame is None:
        print("Failed to read frame.")
        break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)

    # Detect pose and hand landmarks
    pose_result = pose_detector.detect(mp_image)
    hand_result = hand_detector.detect(mp_image)

    # Annotate frame with pose and hand landmarks
    annotated_image = draw_pose_landmarks(image, pose_result)
    annotated_image = draw_hand_landmarks(annotated_image, hand_result)

    currentTime = time.time()
    fps = 1 / (currentTime - previousTime)
    previousTime = currentTime

    cv2.putText(annotated_image, f"{int(fps)} FPS", (10, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
    annotated_bgr_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
    cv2.imshow("Pose and Hand Detection", annotated_bgr_image)

    outVideo.write(frame)
    outResult.write(annotated_bgr_image)

    if cv2.waitKey(5) & 0xFF == ord("q"):
        break

capture.release()
outVideo.release()
outResult.release()
cv2.destroyAllWindows()
