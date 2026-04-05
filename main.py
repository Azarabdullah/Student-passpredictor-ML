from sklearn.linear_model import LogisticRegression
import numpy as np

# Study hours (input)
X = np.array([[1], [2], [3], [4], [5], [6]])

# Result (0 = Fail, 1 = Pass)
y = np.array([0, 0, 0, 1, 1, 1])

# Create model
model = LogisticRegression()

# Train model
model.fit(X, y)

# Predict
hours = [[3.5]]
prediction = model.predict(hours)

if prediction[0] == 1:
    print("Pass")
else:
    print("Fail")
