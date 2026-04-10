import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib

# Load the data
df = pd.read_csv('data/customer_churn.csv')

# Features / target
X = df[['Age', 'Account_Manager', 'Years', 'Num_Sites']]
y = df['Churn']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 🔥 Pipeline = scaling + model
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(random_state=42))
])

# Train
pipeline.fit(X_train, y_train)

# Predict
y_pred = pipeline.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Save pipeline complet (IMPORTANT)
joblib.dump(pipeline, 'data/churn_model_clean.pkl')