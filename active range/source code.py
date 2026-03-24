import cv2
import numpy as np
import time
from google.colab.patches import cv2_imshow

# ---------------- USER INPUT ----------------
save_option = input("Do you want to save the output video? (y/n): ").strip().lower()

# Accept both 'y' and 'yes'
save_output = save_option in ['y', 'yes']

# ---------------- PARAMETERS ----------------
KNOWN_WIDTH = 14.0   # Average face width in cm
FOCAL_LENGTH = 600   # Replace with calibrated value for accuracy

# ---------------- LOAD FACE DETECTOR ----------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

if face_cascade.empty():
    print("Error loading Haar Cascade!")
    exit()

# ---------------- VIDEO INPUT ----------------
video_path = "/content/Range.mp4"  # Upload your video to Colab first

cap = cv2.VideoCapture("/content/1900-151662242_medium.mp4")

if not cap.isOpened():
    print("Error opening video file")
    exit()

# ---------------- PROCESSING SIZE ----------------
PROCESS_WIDTH = 640
PROCESS_HEIGHT = 480

# ---------------- VIDEO WRITER ----------------
out = None
if save_output:
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('/content/output.avi', fourcc, 20.0,
                          (PROCESS_WIDTH, PROCESS_HEIGHT))

# ---------------- DISTANCE FUNCTION ----------------
def find_distance(perceived_width):
    if perceived_width == 0:
        return 0
    return (KNOWN_WIDTH * FOCAL_LENGTH) / perceived_width

# ---------------- SMOOTHING VARIABLE ----------------
prev_distance = 0

# ---------------- MAIN LOOP ----------------
while True:
    ret, frame = cap.read()

    if not ret:
        # Restart video automatically
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    # -------- STEP 1: RESIZE --------
    frame = cv2.resize(frame, (PROCESS_WIDTH, PROCESS_HEIGHT))

    # -------- STEP 2: FACE DETECTION --------
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    if len(faces) > 0:
        # -------- STEP 3: TAKE LARGEST FACE --------
        faces = sorted(faces, key=lambda x: x[2], reverse=True)
        (x, y, w, h) = faces[0]

        # -------- STEP 4: DISTANCE ESTIMATION --------
        distance = find_distance(w)

        # -------- STEP 5: SMOOTHING --------
        distance = 0.7 * prev_distance + 0.3 * distance
        prev_distance = distance

        # -------- DRAW --------
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"Distance: {round(distance, 2)} cm",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2)

    # -------- STEP 6: DISPLAY (COLAB FIX) --------
    cv2_imshow(frame)
    time.sleep(0.03)

    # -------- STEP 7: SAVE --------
    if save_output and out is not None:
        out.write(frame)

    # -------- EXIT CONDITION --------
    # Press stop button in Colab to exit manually

# ---------------- CLEANUP ----------------
cap.release()

if out is not None:
    out.release()

cv2.destroyAllWindows()