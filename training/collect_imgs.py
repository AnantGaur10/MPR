import os
import cv2
import mediapipe as mp
import numpy as np
import time  # added for delay


DATA_DIR = './data_landmarks'  # save landmarks
IMAGE_REF_DIR = './data_reference_images'  # new folder for reference images
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(IMAGE_REF_DIR, exist_ok=True)

number_of_classes = 29
dataset_size = 200

# ------------------------------
# Mediapipe Hands setup
# ------------------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ------------------------------
# Open webcam
# ------------------------------
cap = None
for i in range(5):
    temp_cap = cv2.VideoCapture(i)
    if temp_cap.isOpened():
        cap = temp_cap
        print(f"Using camera index {i}")
        break
    temp_cap.release()

if cap is None:
    print("No camera found. Exiting...")
    exit()

# ------------------------------
# Determine starting class index
# ------------------------------
existing_classes = [
    int(d) for d in os.listdir(DATA_DIR)
    if os.path.isdir(os.path.join(DATA_DIR, d)) and d.isdigit()
]
start_class_idx = max(existing_classes) + 1 if existing_classes else 0

# ------------------------------
# Data collection loop
# ------------------------------
for class_idx in range(start_class_idx, number_of_classes):
    class_dir = os.path.join(DATA_DIR, str(class_idx))
    os.makedirs(class_dir, exist_ok=True)

    image_ref_path = os.path.join(IMAGE_REF_DIR, f'class_{class_idx}.jpg')

    print(f"\nCollecting data for class {class_idx}. Press 'Q' when ready.")

    # Ready screen
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        cv2.putText(frame, 'Ready? Press "Q"', (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1.3, (0, 255, 0), 3)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # ------------------------------
    # Take screenshot after 1 second
    # ------------------------------
    print("Capturing reference image in 1 second...")
    time.sleep(1)
    ret, frame = cap.read()
    if ret:
        frame = cv2.flip(frame, 1)
        cv2.imwrite(image_ref_path, frame)
        print(f"Saved reference image for class {class_idx} → {image_ref_path}")
    else:
        print("Failed to capture reference image.")

    # ------------------------------
    # Start collecting landmarks
    # ------------------------------
    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            data_aux = []
            for lm in hand.landmark:
                data_aux.append(lm.x)
                data_aux.append(lm.y)

            # Pad for consistency (even if only one hand)
            if len(results.multi_hand_landmarks) == 1:
                data_aux.extend([0] * 42)

            np.save(os.path.join(class_dir, f'{counter}.npy'), np.array(data_aux))
            counter += 1
            print(f"Saved {counter}/{dataset_size} for class {class_idx}")

        cv2.imshow('frame', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

# ------------------------------
# Cleanup
# ------------------------------
cap.release()
cv2.destroyAllWindows()
hands.close()
print("Data collection completed.")
