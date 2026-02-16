import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ------------------------------
# Load dataset
# ------------------------------
data_dict = pickle.load(open('./data.pickle', 'rb'))

data = np.array(data_dict['data'])      # already 42 features per sample
labels = np.array(data_dict['labels'])

print("Data shape:", data.shape)  # (num_samples, 42)

# ------------------------------
# Split into train and test sets
# ------------------------------
x_train, x_test, y_train, y_test = train_test_split(
    data, labels,
    test_size=0.2,
    shuffle=True,
    stratify=labels,
    random_state=42
)

# ------------------------------
# Train RandomForest classifier
# ------------------------------
model = RandomForestClassifier(
    n_estimators=200,   # more trees for better performance
    max_depth=None,
    random_state=42
)
model.fit(x_train, y_train)

# ------------------------------
# Evaluate model
# ------------------------------
y_predict = model.predict(x_test)
score = accuracy_score(y_test, y_predict)
print(f"Accuracy: {score * 100:.2f}%")

# ------------------------------
# Save trained model
# ------------------------------
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)

print("Model saved as 'model.p'")
