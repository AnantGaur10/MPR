import os
import numpy as np
import pickle

# ------------------------------
# Dataset paths
# ------------------------------
DATA_DIR = './data_landmarks'  # folder with .npy landmark files
data = []
labels = []

# ------------------------------
# Load landmark files and create dataset
# ------------------------------
for class_dir in os.listdir(DATA_DIR):
    class_path = os.path.join(DATA_DIR, class_dir)
    if not os.path.isdir(class_path):
        continue

    for file_name in os.listdir(class_path):
        file_path = os.path.join(class_path, file_name)
        sample = np.load(file_path)

        # Take only first hand (42 features)
        if len(sample) >= 42:
            sample_42 = sample[:42]  # take first 42 features

            # Split into x and y
            xs = sample_42[0::2]
            ys = sample_42[1::2]

            # Normalize relative to wrist (index 0)
            x0, y0 = xs[0], ys[0]
            xs = [x - x0 for x in xs]
            ys = [y - y0 for y in ys]

            normalized = []
            for i in range(len(xs)):
                normalized.append(xs[i])
                normalized.append(ys[i])

            # Add normalized sample
            data.append(normalized)
            labels.append(int(class_dir))

            # ------------------------------
            # Add mirrored version (flip horizontally)
            # ------------------------------
            mirrored = []
            for i in range(len(xs)):
                mirrored.append(-xs[i])  # flip X-axis
                mirrored.append(ys[i])   # keep Y-axis same

            data.append(mirrored)
            labels.append(int(class_dir))

        else:
            print(f"Skipping {file_path}: not enough features ({len(sample)})")

# ------------------------------
# Save dataset as pickle
# ------------------------------
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print(f"Dataset ready: {len(data)} samples, {len(set(labels))} classes")
