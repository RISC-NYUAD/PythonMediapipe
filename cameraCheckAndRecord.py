import cv2
import time

def initialize_camera(index):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        print(f"Failed to open video source {index}")
        return None

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(3, 1280)
    cap.set(4, 720)
    
    return cap


camera_indices = [0, 1, 2]  
cameras = [initialize_camera(idx) for idx in camera_indices]
cameras = [cam for cam in cameras if cam is not None]

if not cameras:
    print("No cameras available.")
    exit()

source_fps = int(cameras[0].get(cv2.CAP_PROP_FPS))
source_width = int(cameras[0].get(cv2.CAP_PROP_FRAME_WIDTH))
source_height = int(cameras[0].get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Source video FPS: {source_fps}, Dimensions: {source_width}x{source_height}")

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
outResults = [cv2.VideoWriter(f"result_cam{idx}.mp4", fourcc, source_fps, (source_width, source_height)) 
              for idx in range(len(cameras))]

previousTime = time.time()

while True:
    for i, cap in enumerate(cameras):
        ret, frame = cap.read()
        if not ret or frame is None:
            print(f"Failed to read frame from camera {i}.")
            continue

        currentTime = time.time()
        fps = 1 / (currentTime - previousTime)
        previousTime = currentTime

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2.putText(image, f"{int(fps)} FPS", (10, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        annotated_bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imshow(f"Camera {i}", annotated_bgr_image)
        outResults[i].write(annotated_bgr_image)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

for cap in cameras:
    cap.release()
for out in outResults:
    out.release()
cv2.destroyAllWindows()
