import cv2
import time

capture = cv2.VideoCapture(0)

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

    currentTime = time.time()
    fps = 1 / (currentTime - previousTime)
    previousTime = currentTime

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    cv2.putText(image, f"{int(fps)} FPS", (10, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
    annotated_bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imshow("Camera Test", annotated_bgr_image)

    outResult.write(annotated_bgr_image)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

outResult.release()
capture.release()
cv2.destroyAllWindows()
