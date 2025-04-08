import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import xgboost as xgb
import lightgbm as lgb

# Percorso della cartella dei dati
base_folder = "C:\\Users\\Mattia\\Desktop\\Smart Werables\\Project\\data_collection"
subjects = ["Andrea", "Diego", "Fabian", "Giada", "Giulio", "Gugliemo", "Kim", "Lenne", "Leonardo", "Letizia","Mattia", "Nicolas", "Pietro", "Rami", "Simon"]
csv_files = ["0.csv", "1.csv", "2.csv", "3.csv", "4.csv", "5.csv", "6.csv", "7.csv"]

# Mappatura delle etichette
label_map = {0: "Normal Standing", 1: "Heel Tapping", 2: "Toe Tapping", 3: "Standing in one foot", 
             4: "Sitting", 5: "Standing on the other foot", 6: "Walking", 7: "Jumping", 8: "Movment not classified"}

# Funzione per caricare i dati
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

    # Mappatura inversa per riconvertire le etichette in numeri
    inverse_label_map = {v: k for k, v in label_map.items()}
    df["label"] = df["label"].map(inverse_label_map)  # Converti le etichette in numeri

    df.to_csv("C:\\Users\\Mattia\\Desktop\\Smart Werables\\Project\\data_collection\\all_data_labeled.csv", index=False)

    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df.drop(columns=['Timestamp'], inplace=True)

    # Suddivisione in training (70%), validation (15%) e test (15%)
    df_train, df_temp = train_test_split(df, test_size=0.3, stratify=df["label"], random_state=42)
    df_val, df_test = train_test_split(df_temp, test_size=0.5, stratify=df_temp["label"], random_state=42)
    
    X_train, y_train = df_train[['Sensor1', 'Sensor2']], df_train['label']
    X_val, y_val = df_val[['Sensor1', 'Sensor2']], df_val['label']
    X_test, y_test = df_test[['Sensor1', 'Sensor2']], df_test['label']

    # Normalizzazione
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    # Definizione degli iperparametri per la ricerca
    param_grid_xgb = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 6, 9],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.7, 0.8, 1.0],
        'colsample_bytree': [0.7, 0.8, 1.0]
    }
    
    param_grid_lgb = {
        'n_estimators': [100, 200, 300],
        'max_depth': [-1, 10, 20],
        'learning_rate': [0.01, 0.1, 0.2],
        'num_leaves': [20, 31, 40],
        'subsample': [0.7, 0.8, 1.0],
        'colsample_bytree': [0.7, 0.8, 1.0]
    }

    # XGBoost Model
    model_xgb = xgb.XGBClassifier(objective='multi:softmax', num_class=len(label_map), random_state=42)
    grid_search_xgb = RandomizedSearchCV(model_xgb, param_grid_xgb, cv=3, n_iter=10, n_jobs=-1, verbose=2, random_state=42)
    grid_search_xgb.fit(X_train, y_train)
    best_model_xgb = grid_search_xgb.best_estimator_

    print("Migliori iperparametri XGBoost:", grid_search_xgb.best_params_)

    # LightGBM Model
    model_lgb = lgb.LGBMClassifier(objective='multiclass', num_class=len(label_map), random_state=42)
    grid_search_lgb = RandomizedSearchCV(model_lgb, param_grid_lgb, cv=3, n_iter=10, n_jobs=-1, verbose=2, random_state=42)
    grid_search_lgb.fit(X_train, y_train)
    best_model_lgb = grid_search_lgb.best_estimator_

    print("Migliori iperparametri LightGBM:", grid_search_lgb.best_params_)

    # Valutazione sui set di validazione e test
    models = {"XGBoost": best_model_xgb, "LightGBM": best_model_lgb}
    
    print(f"Training Accuracy (XGBoost): {accuracy_score(y_train, best_model_xgb.predict(X_train)):.4f}")
    print(f"Validation Accuracy (XGBoost): {accuracy_score(y_val, best_model_xgb.predict(X_val)):.4f}")
    print(f"Test Accuracy (XGBoost): {accuracy_score(y_test, best_model_xgb.predict(X_test)):.4f}")

    print(f"Training Accuracy (LightGBM): {accuracy_score(y_train, best_model_lgb.predict(X_train)):.4f}")
    print(f"Validation Accuracy (LightGBM): {accuracy_score(y_val, best_model_lgb.predict(X_val)):.4f}")
    print(f"Test Accuracy (LightGBM): {accuracy_score(y_test, best_model_lgb.predict(X_test)):.4f}")

    output_folder = "C:\\Users\\Mattia\\Desktop\\Smart Werables\\Project\\data_collection\\output_graphs_XGB_LGB"
    os.makedirs(output_folder, exist_ok=True)

    for name, model in models.items():
        print(f"\nModello: {name}")
        y_val_pred = best_model_xgb.predict(X_val)
        print(f"Validation Accuracy ({name}):", accuracy_score(y_val, y_val_pred))
        print(f"Validation Classification Report ({name}):\n", classification_report(y_val, y_val_pred))

        y_test_pred = best_model_lgb.predict(X_test)
        print(f"Test Accuracy ({name}):", accuracy_score(y_test, y_test_pred))
        print(f"Test Classification Report ({name}):\n", classification_report(y_test, y_test_pred))

        # Matrice di confusione
        conf_matrix = confusion_matrix(y_test, y_test_pred, labels=np.unique(y_test))
        plt.figure(figsize=(10, 10))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
        plt.xlabel("Predicted Activity")
        plt.ylabel("True Activity")
        plt.title(f"Confusion Matrix - {name}")
        plt.savefig(os.path.join(output_folder, f"confusion_matrix_{name}.png"), dpi=300)
        plt.show()

        
    # Test separato per ogni soggetto
    for subject in subjects:
        print(f"\nTesting per {subject}:")
        df_test = df[df["Subject"] == subject]
        if df_test.empty:
            print(f"Nessun dato trovato per {subject}.")
            continue

        X_test = df_test[['Sensor1', 'Sensor2']]
        y_test = df_test['label']
        X_test = scaler.transform(X_test)
        y_pred = best_model_lgb.predict(X_test)

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
    
    # Salva i modelli e lo scaler
    joblib.dump(best_model_xgb, 'xgboost_model_optimized.pkl')
    joblib.dump(best_model_lgb, 'lightgbm_model_optimized.pkl')
    joblib.dump(scaler, 'scaler_optimized_xgb.pkl')
    print("Modelli ottimizzati e scaler salvati con successo!")
