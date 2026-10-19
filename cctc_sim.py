import cv2

# Paths to your 5 pre-recorded videos (59 sec each)
video_paths = [
    'feed/video3_1.avi',
    'feed/video3_2.avi',
    'feed/video3_3.avi',
    'feed/video3_5.avi'
]

# Create VideoCapture objects
caps = [cv2.VideoCapture(path) for path in video_paths]


def capture_fps(cap):
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if fps < 1.0:
        return None
    return fps

def read_frame(cap, path):
    ret, frame = cap.read()
    
    # If video ends, restart it
    if not ret:
        cap.release()
        cap = cv2.VideoCapture(path)
        ret, frame = cap.read()
    
    return cap, frame


valid_fps = [fps for fps in (capture_fps(cap) for cap in caps) if fps is not None]
frame_delay_ms = max(1, int(round(1000.0 / min(valid_fps)))) if valid_fps else 33

while True:
    frames = []
    
    # Read frames from all cameras
    for i in range(4):
        caps[i], frame = read_frame(caps[i], video_paths[i])
        
        if frame is None:
            frame = cv2.imread("black.jpg")  # fallback
        
        frame = cv2.resize(frame, (320, 240))
        frames.append(frame)

    # Create grid layout (2x3, last one blank)
    row1 = cv2.hconcat(frames[:2])
    row2 = cv2.hconcat(frames[2:])  # duplicate to fill grid

    grid = cv2.vconcat([row1, row2])

    cv2.imshow("5 Camera CCTV Simulation", grid)

    if cv2.waitKey(frame_delay_ms) & 0xFF == ord('q'):
        break

# Release all
for cap in caps:
    cap.release()

cv2.destroyAllWindows()
