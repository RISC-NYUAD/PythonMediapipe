# credits to https://github.com/TemugeB for their amazing work

import cv2
import numpy as np
import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
from utils import DLT, get_projection_matrix

VisionRunningMode = mp.tasks.vision.RunningMode
pose_options = vision.PoseLandmarkerOptions(
    base_options=python.BaseOptions(
        model_asset_path="models/pose_landmarker_full.task",
        delegate=python.BaseOptions.Delegate.CPU
    ),
    running_mode=VisionRunningMode.VIDEO,
    output_segmentation_masks=True,

    min_pose_detection_confidence=0.8,
    min_pose_presence_confidence=0.8,
    min_tracking_confidence=0.8,
)
pose_detector = vision.PoseLandmarker.create_from_options(pose_options)


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
                color=(255, 0, 0), thickness=2, circle_radius=0
            ),
            solutions.drawing_utils.DrawingSpec(
                color=(0, 255, 0), thickness=2, circle_radius=2
            ),
        )
    return annotated_frame


def initialize_cameras(index0=0, index1=1):
    cap0 = cv2.VideoCapture(index0)
    if not cap0.isOpened():
        print(f"Failed to open video source {index0}")
        cap0.release()
        return None

    cap0.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap0.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
    cap0.set(cv2.CAP_PROP_FPS, 30)
    cap0.set(3, 1280)
    cap0.set(4, 720)

    cap1 = cv2.VideoCapture(index1)
    if not cap1.isOpened():
        print(f"Failed to open video source {index1}")
        cap1.release()
        return None

    cap1.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap1.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
    cap1.set(cv2.CAP_PROP_FPS, 30)
    cap1.set(3, 1280)
    cap1.set(4, 720)

    source_fps = int(cap0.get(cv2.CAP_PROP_FPS))
    source_width = int(cap0.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(cap0.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(
        f"Source 0 FPS: {source_fps}, Dimensions: {source_width}x{source_height}")

    source_fps = int(cap1.get(cv2.CAP_PROP_FPS))
    source_width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(
        f"Source 1 FPS: {source_fps}, Dimensions: {source_width}x{source_height}")

    return cap0, cap1


cameras = initialize_cameras(0, 1)

if not cameras:
    print("No cameras available.")
    exit()


previousTime = time.time()


P0 = get_projection_matrix(0)
P1 = get_projection_matrix(1)

kpts_cam0 = []
kpts_cam1 = []
kpts_3d = []

while True:
    ret0, frame0 = cameras[0].read()
    ret1, frame1 = cameras[1].read()

    if not ret0 or not ret1:
        print("Failed to read frames.")
        break

    frame0.flags.writeable = False
    frame1.flags.writeable = False

    frame0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2RGB)
    frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)

    mp_frame0 = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame0)
    mp_frame1 = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame1)

    timestamp = int(time.time() * 1000)
    pose_result0 = pose_detector.detect_for_video(mp_frame0, timestamp)
    timestamp = int(time.time() * 1000)
    pose_result1 = pose_detector.detect_for_video(mp_frame1, timestamp)

    frame0.flags.writeable = True
    frame1.flags.writeable = True

    frame0 = cv2.cvtColor(frame0, cv2.COLOR_RGB2BGR)
    frame1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2BGR)

    frame0_keypoints = []
    if pose_result0.pose_landmarks:
        for pose_landmarks in pose_result0.pose_landmarks:
            for landmark in pose_landmarks:  # Assuming pose_landmarks is a list of NormalizedLandmark objects
                pxl_x = landmark.x * frame0.shape[1]
                pxl_y = landmark.y * frame0.shape[0]
                pxl_x = int(round(pxl_x))
                pxl_y = int(round(pxl_y))
                cv2.circle(frame0, (pxl_x, pxl_y), 5, (0, 0, 255), -1)
                kpts = [pxl_x, pxl_y]
                frame0_keypoints.append(kpts)
    else:
        frame0_keypoints = [[-1, -1]]*33

    kpts_cam0.append(frame0_keypoints)

    frame1_keypoints = []
    if pose_result1.pose_landmarks:
        for pose_landmarks in pose_result1.pose_landmarks:
            for landmark in pose_landmarks:
                pxl_x = landmark.x * frame1.shape[1]
                pxl_y = landmark.y * frame1.shape[0]
                pxl_x = int(round(pxl_x))
                pxl_y = int(round(pxl_y))
                cv2.circle(frame1, (pxl_x, pxl_y), 5, (0, 0, 255), -1)
                kpts = [pxl_x, pxl_y]
                frame1_keypoints.append(kpts)
    else:
        frame1_keypoints = [[-1, -1]]*33

    kpts_cam1.append(frame1_keypoints)

    frame_p3ds = []
    for uv1, uv2 in zip(frame0_keypoints, frame1_keypoints):
        if uv1[0] == -1 or uv2[0] == -1:
            _p3d = [-1, -1, -1]
        else:
            _p3d = DLT(P0, P1, uv1, uv2)
        frame_p3ds.append(_p3d)

    frame_p3ds = np.array(frame_p3ds).reshape((33, 3))
    kpts_3d.append(frame_p3ds)

    frame0 = draw_pose_landmarks(frame0, pose_result0)
    frame1 = draw_pose_landmarks(frame1, pose_result1)

    currentTime = time.time()
    fps = 1 / (currentTime - previousTime)
    previousTime = currentTime

    cv2.putText(frame0, f"{int(fps)} FPS", (10, 70),
                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Dual Cam Pose Detection, Cam 0", frame0)
    cv2.putText(frame1, f"{int(fps)} FPS", (10, 70),
                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Dual Cam Pose Detection, Cam 1", frame1)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cameras[0].release()
cameras[1].release()
cv2.destroyAllWindows()
