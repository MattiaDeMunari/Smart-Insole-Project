import os 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Path to the main folder where subject data is saved
base_folder = "C:\\Users\\Mattia\\Desktop\\Smart Werables\\Project\\data_collection"
subjects = ["Andrea", "Diego" , "Fabian", "Giada" , "Giulio", "Gugliemo" , "Kim", "Lenne" , "Leonardo" , "Letizia" ,"Mattia", "Nicolas" , "Pietro","Rami", "Simon"]
csv_files = ["0.csv", "1.csv", "2.csv", "3.csv", "4.csv", "5.csv", "6.csv", "7.csv"]

# Dictionary to map numerical labels to tasks
label_map = {
    0: "Normal Standing",
    1: "Heel Tapping",
    2: "Toe Tapping",
    3: "Standing in one foot",
    4: "Sitting",
    5: "Standing on the other foot",
    6: "Walking",
    7: "Jumping", 
    8: "Movment not classified"
}

reverse_label_map = {v: k for k, v in label_map.items()}

def load_data_all_subjects():
    df_list = []
    for subject in subjects:
        for file in csv_files:
            file_path = os.path.join(base_folder, subject, file)
            if os.path.exists(file_path):
                temp_df = pd.read_csv(file_path, sep=',')
                temp_df["Subject"] = subject  
                df_list.append(temp_df)
    return pd.concat(df_list, ignore_index=True) if df_list else None

df = load_data_all_subjects()
if df is not None:
    df["label"] = df["label"].map(label_map)
    df.to_csv("C:\\Users\\Mattia\\Desktop\\Smart Werables\\Project\\data_collection\\all_data_labeled.csv", index=False)
    print("File aggiornato con nomi delle attività salvato con successo!")

    # Pre-processing of data
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df.drop(columns=['Timestamp'], inplace=True)
    X = df[['Sensor1', 'Sensor2']]
    y = df['label']
    
    # Division into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Split into formation and validation set for early shutdown
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Finding the optimal number of trees with validation set
    best_n_estimators = 10
    best_accuracy = 0
    accuracies = []

    for n in range(10, 300, 10):  # Testing from 10 to 300 trees
        model = RandomForestClassifier(n_estimators=n, max_depth=10, random_state=42)
        model.fit(X_train, y_train)
        y_val_pred = model.predict(X_val)
        acc = accuracy_score(y_val, y_val_pred)
        accuracies.append(acc)

        # Simulates early stopping
        if acc > best_accuracy:
            best_accuracy = acc
            best_n_estimators = n
        elif acc < best_accuracy - 0.01:  # If accuracy decreases significantly, stop training
            print(f"Stopping early at {n} estimators")
            break

    print(f"Best number of estimators: {best_n_estimators}")

    # Apply the best model to the test set
    final_model = RandomForestClassifier(n_estimators=best_n_estimators, max_depth=10, random_state=42)
    final_model.fit(X_train, y_train)
    y_pred = final_model.predict(X_test)

    y_train_pred = final_model.predict(X_train)
    train_acc = accuracy_score(y_train, y_train_pred)
    print(f"Training Accuracy: {train_acc}")
    print(f"Test Accuracy: {accuracy_score(y_test, y_pred)}")


    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

     # Creating folder to save charts and matrix confusion
    output_folder = "C:\\Users\\Mattia\\Desktop\\Smart Werables\\Project\\data_collection\\output_graphs_RF"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    conf_matrix = confusion_matrix(y_test, y_pred, labels=np.unique(y_test))
    plt.figure(figsize=(10, 10))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
    plt.xlabel("Predicted Activity")
    plt.ylabel("True Activity")
    plt.title("Confusion Matrix - All Subjects")
    plt.savefig(os.path.join(output_folder, "confusion_matrix_all.png"), dpi=300)
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
        y_pred = final_model.predict(X_test)

        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("Classification Report:\n", classification_report(y_test, y_pred))
        conf_matrix = confusion_matrix(y_test, y_pred, labels=np.unique(y_test))
        plt.figure(figsize=(10, 10))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
        plt.xlabel("Predicted Activity")
        plt.ylabel("True Activity")
        plt.title(f"Confusion Matrix - {subject}")
        plt.savefig(os.path.join(output_folder,f"confusion_matrix_{subject}.png"), dpi=300)
        plt.show()

    # Save the model and scaler
    joblib.dump(final_model, 'random_forest_model.pkl')
    joblib.dump(scaler, 'random_forest_scaler.pkl')
    print("Modello e scaler salvati con successo!")
