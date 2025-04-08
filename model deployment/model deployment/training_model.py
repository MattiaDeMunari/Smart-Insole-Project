import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer
import joblib

# Load the CSV files
file_path1 = "./saving_data/training_sensor_data_pressed.csv"
file_path2 = "./saving_data/training_sensor_data_not_pressed.csv"

df1 = pd.read_csv(file_path1)
df2 = pd.read_csv(file_path2)

# Convert "Sensor Value" to numeric
df1["Sensor Value"] = pd.to_numeric(df1["Sensor Value"], errors='coerce')
df2["Sensor Value"] = pd.to_numeric(df2["Sensor Value"], errors='coerce')

# Encode labels: "pressed" -> 1, "not pressed" -> 0
df1["label"] = df1["label"].map({"pressed": 1})
df2["label"] = df2["label"].map({"not_pressed": 0})

# Combine both datasets
df = pd.concat([df1, df2], ignore_index=True)

# Drop Timestamp
df.drop(columns=["Timestamp"], inplace=True)


# Sliding window function
def sliding_window(data, window_length, step_size):
    X, y = [], []
    for i in range(0, len(data) - window_length + 1, step_size):
        window = data.iloc[i:i + window_length]
        # X.append(window["Sensor Value"].values)
        mean = np.mean(window["Sensor Value"])
        std = np.std(window["Sensor Value"])
        min = np.min(window["Sensor Value"])
        max = np.max(window["Sensor Value"])
        ptp = np.ptp(window["Sensor Value"])
        rms = np.sqrt(np.mean(window["Sensor Value"] ** 2))
        # np.count_nonzero(np.diff(np.sign(window["Sensor Value"]), axis=1), axis=1)]
        # Store extracted features
        X.append([mean, std, min, max, ptp, rms])
        y.append(window["label"].values[-1])
    return np.array(X), np.array(y)


# Apply sliding window
X, y = sliding_window(df, window_length=20, step_size=10)

# Handle NaN values
imputer = SimpleImputer(strategy="mean")
X = imputer.fit_transform(X)

# Split data into train (70%), validation (15%), and test (15%)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# imputer = SimpleImputer(strategy="mean")
# X_train = imputer.fit_transform(X_train)  # Fit on training only
# X_val = imputer.transform(X_val)
# X_test = imputer.transform(X_test)

# Create a pipeline with standardization and SVM classifier
svm_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42))
])

# Train the SVM model
svm_pipeline.fit(X_train, y_train)

# Predict on validation and test sets
y_val_pred = svm_pipeline.predict(X_val)
y_test_pred = svm_pipeline.predict(X_test)

# Evaluate model performance
val_accuracy = accuracy_score(y_val, y_val_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)
classification_rep = classification_report(y_test, y_test_pred)

# Print results
print(f"Validation Accuracy: {val_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
print("Classification Report:\n", classification_rep)

# Save the trained SVM model
joblib.dump(svm_pipeline, "svm_model.pkl")
print("Model saved as 'svm_model.pkl'")
