import csv
import numpy as np
import pickle
import os
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# gesture labels
GESTURES = {
    '0': 'Open Palm',
    '1': 'Fist',
    '2': 'Point Up',
    '3': 'Point Down',
    '4': 'Point Left',
    '5': 'Point Right',
}

print("Loading data...")

X = []
y = []

with open('data/keypoints.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        if not row:
            continue
        y.append(int(row[0]))
        X.append([float(v) for v in row[1:]])

X = np.array(X)
y = np.array(y)

print(f"Loaded {len(X)} samples across {len(set(y))} gestures\n")

# split into train and test sets (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples\n")

# train the model
print("Training model...")
clf = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    max_iter=1000,
    random_state=42
)
clf.fit(X_train, y_train)
print("Training complete!\n")

# evaluate
y_pred = clf.predict(X_test)

print("--- Results ---")
print(classification_report(
    y_test, y_pred,
    target_names=[GESTURES[str(i)] for i in sorted(set(y))]
))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print()

# save model
os.makedirs('model', exist_ok=True)
with open('model/gesture_model.pkl', 'wb') as f:
    pickle.dump(clf, f)

print("Model saved to model/gesture_model.pkl")
print("Copy this file to your drone/ folder when ready")
