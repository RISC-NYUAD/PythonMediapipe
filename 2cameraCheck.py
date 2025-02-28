import cv2
import time

capture0 = cv2.VideoCapture(1)
capture1 = cv2.VideoCapture(0)

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

source_fps_capture0 = int(capture0.get(cv2.CAP_PROP_FPS))
source_width_capture0 = int(capture0.get(cv2.CAP_PROP_FRAME_WIDTH))
source_height_capture0 = int(capture0.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Source 0 video FPS: {source_fps_capture0}, Dimensions: {source_width_capture0}x{source_height_capture0}")

source_fps_capture1 = int(capture1.get(cv2.CAP_PROP_FPS))
source_width_capture1 = int(capture1.get(cv2.CAP_PROP_FRAME_WIDTH))
source_height_capture1 = int(capture1.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Source 1 video FPS: {source_fps_capture1}, Dimensions: {source_width_capture1}x{source_height_capture1}")

fourcc0 = cv2.VideoWriter_fourcc(*"mp4v")
outResult0 = cv2.VideoWriter("result0.mp4", fourcc0, source_fps_capture0, (source_width_capture0, source_height_capture0))

fourcc1 = cv2.VideoWriter_fourcc(*"mp4v")
outResult1 = cv2.VideoWriter("result1.mp4", fourcc1, source_fps_capture1, (source_width_capture1, source_height_capture1))

previousTime = 0

while capture0.isOpened():
    ret0, frame0 = capture0.read()
    ret1, frame1 = capture1.read()

    if not ret0 or frame0 is None:
        print("Failed to read frame from source 0.")
        break

    if not ret1 or frame1 is None:
        print("Failed to read frame from source 1.")
        break

    currentTime = time.time()
    fps = 1 / (currentTime - previousTime)
    previousTime = currentTime

    image0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2RGB)
    image1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)

    cv2.putText(image0, f"{int(fps)} FPS", (10, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(image1, f"{int(fps)} FPS", (10, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)


    annotated_bgr_image0 = cv2.cvtColor(image0, cv2.COLOR_RGB2BGR)
    annotated_bgr_image1 = cv2.cvtColor(image1, cv2.COLOR_RGB2BGR)

    cv2.imshow("Source 0", annotated_bgr_image0)
    cv2.imshow("Source 1", annotated_bgr_image1)

    outResult0.write(annotated_bgr_image0)
    outResult1.write(annotated_bgr_image1)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

outResult0.release()
outResult1.release()
capture0.release()
capture1.release()
cv2.destroyAllWindows()
