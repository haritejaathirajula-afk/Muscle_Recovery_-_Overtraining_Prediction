import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load your dataset
df = pd.read_csv("Muscle_Recovery_Final_Dataset.csv")

print("Dataset Loaded:\n", df.head())

# Features and Target
X = df[['Training_Load','Sleep_Hours','Stress_Level']]
y = df['Recovery_Status']

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Save model
joblib.dump(model, "model.pkl")

print("Model trained and saved as model.pkl")