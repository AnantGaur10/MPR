import pickle
import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
import os
from threading import Thread, Lock
from collections import deque, Counter

# ------------------------------
# Text-to-Speech (Thread Safe)
# ------------------------------
is_voice_on = True
speech_lock = Lock()

def say_text(text):
    if not is_voice_on or not text.strip():
        return

    def speak():
        try:
            with speech_lock:
                engine = pyttsx3.init()
                voices = engine.getProperty('voices')
                if voices:
                    engine.setProperty('voice', voices[0].id)
                engine.setProperty('rate', 100)
                engine.say(text)
                engine.runAndWait()
                engine.stop()
        except:
            pass

    Thread(target=speak, daemon=True).start()

# ------------------------------
# Load Model
# ------------------------------
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Labels (A–Z, ENTER, SPACE, DELETE)
labels_dict = {i: chr(65 + i) for i in range(26)}
labels_dict[26] = "ENTER"
labels_dict[27] = "SPACE"
labels_dict[28] = "DELETE"        # NEW THUMBS DOWN CLASS

# ------------------------------
# Mediapipe Setup
# ------------------------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.6
)

# ------------------------------
# Extract ROI
# ------------------------------
def extract_hand_roi(frame, hand_landmarks):
    h, w, _ = frame.shape

    xs = [lm.x for lm in hand_landmarks.landmark]
    ys = [lm.y for lm in hand_landmarks.landmark]

    x_min = int(min(xs) * w)
    x_max = int(max(xs) * w)
    y_min = int(min(ys) * h)
    y_max = int(max(ys) * h)

    padding = 40
    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(w, x_max + padding)
    y_max = min(h, y_max + padding)

    return frame[y_min:y_max, x_min:x_max], (x_min, y_min, x_max, y_max)

# ------------------------------
# Prediction Buffer
# ------------------------------
buffer_size = 3
pred_buffer = deque(maxlen=buffer_size)

# ------------------------------
# Main Recognition Loop
# ------------------------------
def recognize():
    global is_voice_on

    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("Camera error")
        return

    text = ""
    word = ""
    sentence = ""

    count_same = 0
    predict_enabled = False
    thresh_display = np.zeros((200, 200), dtype=np.uint8)

    print("\n=== SIGN LANGUAGE RECOGNITION ===")
    print("Q=Quit | V=Voice | SPC=Space | BKSP=Del | ENT=Speak | C=Clear | P=Predict")
    print("=================================\n")

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (640, 480))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        old_text = text
        hand_detected = False
        bbox = None
        pred_conf = 0

        if predict_enabled and results.multi_hand_landmarks:
            hand_detected = True
            hand_landmarks = results.multi_hand_landmarks[0]

            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(255,255,255), thickness=2)
            )

            roi, bbox = extract_hand_roi(frame, hand_landmarks)

            if roi.size > 0:
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                thresh_display = cv2.resize(thresh, (200, 200))

                xs = [lm.x for lm in hand_landmarks.landmark]
                ys = [lm.y for lm in hand_landmarks.landmark]

                wrist_x, wrist_y = xs[0], ys[0]
                xs = [x - wrist_x for x in xs]
                ys = [y - wrist_y for y in ys]

                data_aux = []
                for x, y in zip(xs, ys):
                    data_aux.extend([x, y])

                if len(data_aux) == 42:
                    pred = model.predict([np.asarray(data_aux)])[0]
                    pred_buffer.append(pred)

                    most_common, count = Counter(pred_buffer).most_common(1)[0]
                    predicted_char = labels_dict[int(most_common)]
                    text = predicted_char
                    pred_conf = count / buffer_size

                    # Stabilization
                    if old_text == text:
                        count_same += 1
                    else:
                        count_same = 0

                    # ------------------------------
                    # PERFORM ACTION ON GESTURE
                    # ------------------------------
                    if count_same > 15:
                        if text == "ENTER":
                            if sentence.strip():
                                say_text(sentence)
                                sentence = ""
                                word = ""
                        elif text == "SPACE":
                            if word:
                                sentence += " "
                                word = ""
                        elif text == "DELETE":
                            # --------------------
                            # NEW DELETE GESTURE
                            # deletes one character
                            # --------------------
                            if sentence:
                                sentence = sentence[:-1]
                            if word:
                                word = word[:-1]
                        else:
                            # Normal A–Z
                            if not word or text != word[-1]:
                                word += text
                                sentence += text

                        count_same = 0

        else:
            if not predict_enabled:
                text = ""
                word = ""
                count_same = 0

        # Draw ROI box
        if bbox:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

        # UI Panel
        blackboard = np.zeros((480, 640, 3), dtype=np.uint8)

        cv2.putText(blackboard, "Sign Language Recognition", (120, 40),
                    cv2.FONT_HERSHEY_TRIPLEX, 1.2, (255,165,0), 2)

        cv2.putText(blackboard, f"Prediction: {'ON' if predict_enabled else 'OFF'}",
                    (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0,255,0) if predict_enabled else (0,0,255), 2)

        cv2.putText(blackboard, f"Voice: {'ON' if is_voice_on else 'OFF'}",
                    (500, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0,255,0) if is_voice_on else (0,0,255), 2)

        cv2.putText(blackboard, f"Hand: {'DETECTED' if hand_detected else 'NOT DETECTED'}",
                    (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0,255,0) if hand_detected else (0,0,255), 2)

        cv2.putText(blackboard, f"Pred: {text} ({pred_conf*100:.1f}%)",
                    (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

        cv2.putText(blackboard, "Word:", (30, 230),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)
        cv2.putText(blackboard, word, (30, 270),
                    cv2.FONT_HERSHEY_TRIPLEX, 1.2, (255,255,0), 2)

        cv2.putText(blackboard, "Sentence:", (30, 320),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)

        display_sentence = sentence if len(sentence) < 35 else "..." + sentence[-32:]
        cv2.putText(blackboard, display_sentence, (30, 360),
                    cv2.FONT_HERSHEY_DUPLEX, 1, (255,255,255), 2)

        cv2.putText(
            blackboard,
            "Q:Quit | V:Voice | SPC:Space | BKSP:Del | ENT:Speak | C:Clear | P:Predict",
            (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150,150,150), 1
        )

        combined = np.hstack((frame, blackboard))
        cv2.imshow("Sign Language Recognition", combined)
        cv2.imshow("Threshold View", thresh_display)

        # Keyboard controls
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('v'):
            is_voice_on = not is_voice_on
        elif key == ord(' '):
            sentence += " "
            word = ""
        elif key == 8:
            sentence = sentence[:-1]
            word = ""
        elif key == 13:
            if sentence.strip():
                say_text(sentence)
        elif key == ord('c'):
            sentence = ""
            word = ""
            text = ""
        elif key == ord('p'):
            predict_enabled = not predict_enabled

    cam.release()
    cv2.destroyAllWindows()
    hands.close()

# ------------------------------
# Run
# ------------------------------
if __name__ == "__main__":
    recognize()
