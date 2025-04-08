import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import butter, filtfilt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Path to the main folder where the subject data is saved
base_folder = "C:\\Users\\Mattia\\Desktop\\Smart Werables\\Project\\data_collection"
subjects = ["Andrea", "Diego", "Fabian", "Giada", "Giulio", "Gugliemo", "Kim", "Lenne", "Leonardo", "Letizia","Mattia", "Nicolas", "Pietro", "Rami", "Simon"]
csv_files = ["0.csv", "1.csv", "2.csv", "3.csv", "4.csv", "5.csv", "6.csv", "7.csv"]


# Label Mapping
label_map = {0: "Normal Standing", 1: "Heel Tapping", 2: "Toe Tapping", 3: "Standing in one foot", 
             4: "Sitting", 5: "Standing on the other foot", 6: "Walking", 7: "Jumping", 8: "Movment not classified"}

# Reverse mapping to convert labels back to numbers
inverse_label_map = {v: k for k, v in label_map.items()}

# Function to apply moving average filter
def moving_average(signal, window_size=5):
    return np.convolve(signal, np.ones(window_size)/window_size, mode='same')

# Function to apply Butterworth filter
def butter_lowpass_filter(signal, cutoff=2.5, fs=50, order=3):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, signal)

# Loading data from all subjects
def load_data_all_subjects():
    df_list = []
    for subject in subjects:
        for file in csv_files:
            file_path = os.path.join(base_folder, subject, file)
            if os.path.exists(file_path):
                temp_df = pd.read_csv(file_path, sep=',')
                temp_df["Subject"] = subject  # Aggiunge il nome del soggetto
                df_list.append(temp_df)
    return pd.concat(df_list, ignore_index=True) if df_list else None

df = load_data_all_subjects()
if df is not None:
    df["label"] = df["label"].map(label_map)

    df["label"] = df["label"].map(inverse_label_map)  # Convert labels to numbers

    df["Sensor1"] = moving_average(df["Sensor1"])
    df["Sensor1"] = butter_lowpass_filter(df["Sensor1"])
    df["Sensor2"] = moving_average(df["Sensor2"])
    df["Sensor2"] = butter_lowpass_filter(df["Sensor2"])
    
    df.to_csv("C:\\Users\\Mattia\\Desktop\\Smart Werables\\Project\\data_collection\\filtered_data.csv", index=False)
    

    # Splitting the dataset into training (70%), validation (15%) and test (15%)
    df_train, df_temp = train_test_split(df, test_size=0.3, stratify=df["label"], random_state=42)
    df_val, df_test = train_test_split(df_temp, test_size=0.5, stratify=df_temp["label"], random_state=42)
    
    X_train, y_train = df_train[['Sensor1', 'Sensor2']], df_train['label']
    X_val, y_val = df_val[['Sensor1', 'Sensor2']], df_val['label']
    X_test, y_test = df_test[['Sensor1', 'Sensor2']], df_test['label']
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    # Training the KNN model
    param_grid = {'n_neighbors': [3, 5, 7, 9], 'weights': ['uniform', 'distance'], 'metric': ['euclidean', 'manhattan','minkowski']}
    model = KNeighborsClassifier()
    grid_search = GridSearchCV(model, param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    print("Migliori iperparametri trovati:", grid_search.best_params_)
    
    # Validation set
    y_val_pred = best_model.predict(X_val)
    print("Validation Accuracy:", accuracy_score(y_val, y_val_pred))
    print("Validation Classification Report:\n", classification_report(y_val, y_val_pred))
    
    y_test_pred = best_model.predict(X_test)
    print("Test Accuracy:", accuracy_score(y_test, y_test_pred))
    print("Test Classification Report:\n", classification_report(y_test, y_test_pred))

    # Confusion matrix for testing
    output_folder = "C:\\Users\\Mattia\\Desktop\\Smart Werables\\Project\\data_collection\\output_graphs_KNN_filtered"
    os.makedirs(output_folder, exist_ok=True)
    
    conf_matrix = confusion_matrix(y_test, y_test_pred, labels=np.unique(y_test))
    plt.figure(figsize=(10, 10))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
    plt.xlabel("Predicted Activity")
    plt.ylabel("True Activity")
    plt.title("Confusion Matrix - Test Set")
    plt.savefig(os.path.join(output_folder, "confusion_matrix_test.png"), dpi=300)
    plt.show()

    # Separate test for each subject
    for subject in subjects:
        print(f"\nTesting per {subject}:")
        df_test = df[df["Subject"] == subject]
        if df_test.empty:
            print(f"Nessun dato trovato per {subject}.")
            continue

        X_test = df_test[['Sensor1', 'Sensor2']]
        y_test = df_test['label']
        X_test = scaler.transform(X_test)
        y_pred = best_model.predict(X_test)

        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("Classification Report:\n", classification_report(y_test, y_pred))
        conf_matrix = confusion_matrix(y_test, y_pred, labels=np.unique(y_test))
        plt.figure(figsize=(10, 10))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
        plt.xlabel("Predicted Activity")
        plt.ylabel("True Activity")
        plt.title(f"Confusion Matrix - {subject}")
        plt.savefig(os.path.join(output_folder, f"confusion_matrix_{subject}.png"), dpi=300)
        plt.show()
    
    # Save the KNN model and scaler for future use.
    joblib.dump(best_model, 'knn_model_filtered.pkl')
    joblib.dump(scaler, 'scaler_knn_filtered.pkl')
    print("Modello e scaler salvati con successo!")
