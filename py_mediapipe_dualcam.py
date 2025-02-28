#credits to https://github.com/TemugeB for their amazing work

import cv2
import numpy as np
import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.python.solutions import drawing_utils, pose
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

def _make_homogeneous_rep_matrix(R, t):
    P = np.zeros((4,4))
    P[:3,:3] = R
    P[:3, 3] = t.reshape(3)
    P[3,3] = 1
    return P

def DLT(P1, P2, point1, point2):
    A = [point1[1]*P1[2,:] - P1[1,:],
         P1[0,:] - point1[0]*P1[2,:],
         point2[1]*P2[2,:] - P2[1,:],
         P2[0,:] - point2[0]*P2[2,:]
        ]
    A = np.array(A).reshape((4,4))

    B = A.transpose() @ A
    from scipy import linalg
    U, s, Vh = linalg.svd(B, full_matrices = False)

    return Vh[3,0:3]/Vh[3,3]

def read_camera_parameters(camera_id):
    inf = open('camera_parameters/c' + str(camera_id) + '.dat', 'r')
    cmtx = []
    dist = []
    line = inf.readline()
    for _ in range(3):
        line = inf.readline().split()
        line = [float(en) for en in line]
        cmtx.append(line)
    line = inf.readline()
    line = inf.readline().split()
    line = [float(en) for en in line]
    dist.append(line)
    return np.array(cmtx), np.array(dist)

def read_rotation_translation(camera_id, savefolder = 'camera_parameters/'):
    inf = open(savefolder + 'rot_trans_c'+ str(camera_id) + '.dat', 'r')
    inf.readline()
    rot = []
    trans = []
    for _ in range(3):
        line = inf.readline().split()
        line = [float(en) for en in line]
        rot.append(line)
    inf.readline()
    for _ in range(3):
        line = inf.readline().split()
        line = [float(en) for en in line]
        trans.append(line)
    inf.close()
    return np.array(rot), np.array(trans)

def get_projection_matrix(camera_id):
    cmtx, dist = read_camera_parameters(camera_id)
    rvec, tvec = read_rotation_translation(camera_id)
    P = cmtx @ _make_homogeneous_rep_matrix(rvec, tvec)[:3,:]
    return P


def get_camera_streams():
    cap0 = cv2.VideoCapture(0)
    cap1 = cv2.VideoCapture(1)
    
    for cap in [cap0, cap1]:
        if not cap.isOpened():
            print("Failed to open cameras.")
            exit()
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(3, 1280)
        cap.set(4, 720)
    return cap0, cap1

def process_frame(frame, detector):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    result = detector.detect_for_video(mp_frame, int(time.time() * 1000))
    return result.pose_landmarks

def extract_keypoints(pose_landmarks, frame_shape):
    keypoints = []
    if pose_landmarks:
        for lm in pose_landmarks[0]:
            x, y = int(lm.x * frame_shape[1]), int(lm.y * frame_shape[0])
            keypoints.append([x, y])
    return keypoints if keypoints else [[-1, -1]] * 12

cap0, cap1 = get_camera_streams()
P0, P1 = get_projection_matrix(0), get_projection_matrix(1)

fourcc0 = cv2.VideoWriter_fourcc(*"mp4v")
outResult0 = cv2.VideoWriter("result0.mp4", fourcc0, 30, (1280, 720))

fourcc1 = cv2.VideoWriter_fourcc(*"mp4v")
outResult1 = cv2.VideoWriter("result1.mp4", fourcc1, 30, (1280, 720))

previousTime = 0
while True:
    ret0, frame0 = cap0.read()
    ret1, frame1 = cap1.read()
    
    if not ret0 or not ret1:
        break
    
    cam0_detection_results = process_frame(frame0, pose_detector)
    cam1_detection_results = process_frame(frame1, pose_detector)

    kpts_cam0 = extract_keypoints(cam0_detection_results, frame0.shape)
    kpts_cam1 = extract_keypoints(cam1_detection_results, frame1.shape)
    
    kpts_3d = [DLT(P0, P1, uv1, uv2) if uv1[0] != -1 and uv2[0] != -1 else [-1, -1, -1] 
               for uv1, uv2 in zip(kpts_cam0, kpts_cam1)]
    
    for (x, y) in kpts_cam0:
        if x != -1:
            cv2.circle(frame0, (x, y), 3, (0, 255, 0), 10)
    for (x, y) in kpts_cam1:
        if x != -1:
            cv2.circle(frame1, (x, y), 3, (0, 255, 0), 10)
    
    currentTime = time.time()
    fps = 1 / (currentTime - previousTime)
    previousTime = currentTime

    outResult0.write(frame0)
    outResult1.write(frame1)
    
    cv2.putText(frame0, f"{int(fps)} FPS", (10, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame1, f"{int(fps)} FPS", (10, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Camera 0", frame0)
    cv2.imshow("Camera 1", frame1)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap0.release()
cap1.release()
outResult0.release()
outResult1.release()
cv2.destroyAllWindows()
