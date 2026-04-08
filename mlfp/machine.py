import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load data
df = pd.read_csv('mlfp\machine.csv')

# Preprocessing
print(df.head())
print(df.isna().sum())
print(df.describe())

# Features & target
x = df[['temperature','vibration','pressure','hours_used']]
y = df['failure']

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Model training (no scaling needed)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(x_train, y_train)

# Predictions
y_pred = model.predict(x_test)

# New data prediction
new_data = [[100, 0.09, 27, 1200]]
print("New Data Prediction:", model.predict(new_data))

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Plotting (correct way)
features = ['temperature','vibration','pressure','hours_used']

plt.figure(figsize=(12,8))

for i, feature in enumerate(features):
    plt.subplot(2, 2, i+1)
    plt.scatter(x[feature], y)
    plt.xlabel(feature)
    plt.ylabel("Failure")
    plt.title(f"{feature} vs Failure")

plt.tight_layout()
plt.show()

import joblib
joblib.dump(model, "model.pkl")

