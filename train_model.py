import joblib
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

# Load dữ liệu Iris
iris = load_iris()
X, y = iris.data, iris.target

# Train model
model = LogisticRegression(max_iter=200)
model.fit(X, y)

# Save model
joblib.dump(model, "iris_model.pkl")
print("✅ Model saved as iris_model.pkl")
