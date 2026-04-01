import cv2
import numpy as np
import os

# 1. Path Verification
video_path = r"C:\Users\nithy\OneDrive\Desktop\videoplayback.mp4"
if not os.path.exists(video_path):
    print(f"Error: File not found at {video_path}")
    exit()

cap = cv2.VideoCapture(video_path)
fgbg = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)

class TrackedObject:
    def __init__(self, x, y):
        # 4 state variables (x, y, dx, dy), 2 measurements (x, y)
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                                 [0, 1, 0, 0]], np.float32)
        
        # Transition Matrix (x = x + dx)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                                [0, 1, 0, 1],
                                                [0, 0, 1, 0],
                                                [0, 0, 0, 1]], np.float32)
        
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1
        
        # Initialize state
        initial_state = np.array([[x], [y], [0], [0]], dtype=np.float32)
        self.kalman.statePre = initial_state
        self.kalman.statePost = initial_state
        
        self.predicted = (int(x), int(y))
        self.trace = []
        self.skipped_frames = 0 # To delete old objects

    def predict(self):
        prediction = self.kalman.predict()
        self.predicted = (int(prediction[0]), int(prediction[1]))
        return self.predicted

    def update(self, cx, cy):
        measured = np.array([[np.float32(cx)], [np.float32(cy)]])
        self.kalman.correct(measured)
        self.skipped_frames = 0

# Initialize the list of tracked objects HERE
tracked_objects = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    fgmask = fgbg.apply(frame)
    _, thresh = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detections = []
    for cnt in contours:
        if cv2.contourArea(cnt) > 500:
            x, y, w, h = cv2.boundingRect(cnt)
            detections.append((x + w // 2, y + h // 2, x, y, w, h))

    # 1. Prediction Phase
    for obj in tracked_objects:
        obj.predict()
        obj.skipped_frames += 1

    # 2. Update/Assignment Phase
    for (cx, cy, x, y, w, h) in detections:
        matched = False
        for obj in tracked_objects:
            px, py = obj.predicted
            # Using Euclidean distance is usually more robust than abs difference
            dist = np.sqrt((cx - px)**2 + (cy - py)**2)
            if dist < 60: 
                obj.update(cx, cy)
                matched = True
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                break
        
        if not matched:
            tracked_objects.append(TrackedObject(cx, cy))

    # 3. Drawing and Cleanup Phase
    active_objects = []
    for obj in tracked_objects:
        if obj.skipped_frames < 10:  # Keep object for 10 frames if lost
            active_objects.append(obj)
            # Draw trace
            obj.trace.append(obj.predicted)
            if len(obj.trace) > 20: obj.trace.pop(0)
            
            for i in range(1, len(obj.trace)):
                cv2.line(frame, obj.trace[i - 1], obj.trace[i], (0, 0, 255), 2)
            cv2.circle(frame, obj.predicted, 5, (255, 0, 0), -1)
            
    tracked_objects = active_objects

    cv2.imshow("Kalman Multi-Object Tracking", frame)
    if cv2.waitKey(30) & 0xFF == 27: # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
