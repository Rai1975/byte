import cv2
import mediapipe as mp
from pynput.mouse import Button, Controller
import time

# --- Configuration ---
# Bite detection measures how much the chin-to-nose distance shrinks when you clench.
# Calibrates automatically on startup — keep jaw relaxed during calibration.
GENTLE_BITE_PERCENT = 0.06   # 6% drop from baseline = left click
HARD_BITE_PERCENT = 0.12     # 12% drop from baseline = right click
CALIBRATION_FRAMES = 60      # Frames to average for baseline
COOLDOWN_SECONDS = 0.4       # Minimum time between clicks

# --- Setup ---
mouse = Controller()
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(max_num_faces=1, min_detection_confidence=0.5)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit(1)

# MediaPipe Face Mesh landmarks
CHIN = 152
NOSE = 168

# --- Calibration ---
print("Calibrating... keep your jaw RELAXED for a few seconds.")
baseline_samples = []

while len(baseline_samples) < CALIBRATION_FRAMES:
    ret, frame = cap.read()
    if not ret:
        continue
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        lm = results.multi_face_landmarks[0].landmark
        dist = abs(lm[CHIN].y - lm[NOSE].y)
        baseline_samples.append(dist)

        progress = len(baseline_samples) / CALIBRATION_FRAMES
        h, w, _ = frame.shape
        cv2.putText(frame, f"Calibrating... {int(progress * 100)}%", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
        bar_w = int(progress * 300)
        cv2.rectangle(frame, (10, 50), (10 + bar_w, 70), (0, 200, 255), -1)

    cv2.imshow("Bite Click", frame)
    cv2.waitKey(1)

baseline = sum(baseline_samples) / len(baseline_samples)
print(f"Baseline jaw distance: {baseline:.4f}")
print(f"Gentle bite threshold: {baseline * (1 - GENTLE_BITE_PERCENT):.4f}")
print(f"Hard bite threshold:   {baseline * (1 - HARD_BITE_PERCENT):.4f}")
print("Ready. Bite gently to left-click, bite hard to right-click. Press 'q' to quit.")

# --- Main loop ---
last_click_time = 0
clicked = False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    status = "Relaxed"
    now = time.time()

    if results.multi_face_landmarks:
        lm = results.multi_face_landmarks[0].landmark
        chin = lm[CHIN]
        nose = lm[NOSE]
        dist = abs(chin.y - nose.y)
        drop = (baseline - dist) / baseline

        can_click = (now - last_click_time) > COOLDOWN_SECONDS

        if drop > HARD_BITE_PERCENT and not clicked and can_click:
            mouse.click(Button.right)
            clicked = True
            last_click_time = now
            status = "RIGHT CLICK"
            print("RIGHT CLICK")
        elif drop > GENTLE_BITE_PERCENT and not clicked and can_click:
            mouse.click(Button.left)
            clicked = True
            last_click_time = now
            status = "LEFT CLICK"
            print("LEFT CLICK")
        elif drop <= GENTLE_BITE_PERCENT * 0.5:
            clicked = False

        # Visual feedback
        h, w, _ = frame.shape
        cv2.circle(frame, (int(nose.x * w), int(nose.y * h)), 4, (255, 0, 0), -1)
        cv2.circle(frame, (int(chin.x * w), int(chin.y * h)), 4, (0, 255, 0), -1)
        cv2.line(frame,
                 (int(nose.x * w), int(nose.y * h)),
                 (int(chin.x * w), int(chin.y * h)),
                 (200, 200, 200), 1)

        color = (0, 0, 255) if "CLICK" in status else (200, 200, 200)
        cv2.putText(frame, f"Drop: {drop:.3f} ({drop*100:.1f}%)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, status, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, f"Baseline: {baseline:.4f}  Current: {dist:.4f}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    else:
        cv2.putText(frame, "No face detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow("Bite Click", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Stopped.")
