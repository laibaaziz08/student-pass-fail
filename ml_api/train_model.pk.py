from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

# Inputs: [study hours, previous exam score]
X = np.array([
    [1, 40],
    [2, 45],
    [3, 50],
    [4, 60],
    [5, 65],
    [6, 70],
    [7, 75],
    [8, 85],
    [2, 80],
    [1, 30],
])

# Results: 0 = Fail, 1 = Pass
y = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 0])

model = LogisticRegression()
model.fit(X, y)

# Save the model beside this Python file.
model_path = Path(__file__).resolve().parent / "model.pkl"
joblib.dump(model, model_path)

print("Model trained and saved!")
print(model_path)