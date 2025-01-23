import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  # Für verbesserte Visualisierung
import pandas as pd
import gymnasium as gym
from stable_baselines3 import PPO  # Annahme: Modell ist mit Stable Baselines3 trainiert
from sklearn.preprocessing import MinMaxScaler  # Für die Normalisierung
from sklearn.ensemble import RandomForestClassifier  # Für Feature Importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Pfade für Speicherung
DATA_DIR = "plot_forest"
MODEL_PATH = "models/ppo-LunarLander-v3/best_model.zip"  # Pfad zu deinem vortrainierten Modell
NUM_EPISODES = 100  # Anzahl der Episoden zum Sammeln von Daten

def load_model(model_path):
    """
    Lädt das vortrainierte PPO-Modell.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modell nicht gefunden unter: {model_path}")
    model = PPO.load(model_path)
    return model

def collect_data(model, env_id="LunarLander-v3", num_episodes=100):
    """
    Führt das Modell in der Umgebung aus und sammelt Zustands-Aktions-Paare.
    """
    env = gym.make(env_id)
    data = []
    
    for episode in range(num_episodes):
        state, _ = env.reset(seed=episode)  # Unterschiedliche Seeds für verschiedene Episoden
        done = False
        step = 0  # Schrittzähler
        while not done:
            action, _ = model.predict(state, deterministic=True)
            data.append((state, action))
            state, reward, done, truncated, info = env.step(action)
            step += 1
        print(f"Episoden {episode + 1}/{num_episodes} abgeschlossen, Schritte: {step}")  # Fortschritt anzeigen
    
    env.close()
    return data

def save_data_and_feature_selection(data, labels):
    """
    Speichert die gesammelten Daten und führt die Feature Selection mit Random Forest durch.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Organisiere die Daten in ein DataFrame
    states = np.array([item[0] for item in data])
    actions = np.array([item[1] for item in data])
    df = pd.DataFrame(states, columns=labels)
    df['action'] = actions
    
    # Optional: Sie können hier zusätzliche abgeleitete Features hinzufügen, falls gewünscht.
    # Da Sie sich jedoch auf die originalen Features konzentrieren möchten, werden wir dies überspringen.
    
    # Normalisiere die Features
    scaler = MinMaxScaler()
    feature_columns = [col for col in df.columns if col != 'action']
    df_normalized = df.copy()
    df_normalized[feature_columns] = scaler.fit_transform(df[feature_columns])
    
    # Speichere das normalisierte Datenset als CSV
    df_normalized.to_csv(os.path.join(DATA_DIR, "all_data_normalized.csv"), index=False)
    
    # Feature Selection mit Random Forest
    print("Berechne Feature Importance mit Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(df_normalized.drop(['action'], axis=1), df_normalized['action'])
    feature_importances = pd.Series(rf.feature_importances_, index=df_normalized.drop(['action'], axis=1).columns)
    feature_importances = feature_importances.sort_values(ascending=False)
    print("Feature Importance:")
    print(feature_importances)
    
    # Plot der Feature Importance
    plt.figure(figsize=(12, 8))
    sns.barplot(x=feature_importances.values, y=feature_importances.index, palette='viridis')
    plt.title("Feature Importance aus Random Forest")
    plt.xlabel("Wichtigkeit")
    plt.ylabel("Features")
    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, "RandomForest_Feature_Importance.png"))
    plt.close()
    
    # Speicherung der Feature Importances als CSV
    feature_importances.to_csv(os.path.join(DATA_DIR, "RandomForest_Feature_Importances.csv"))
    print("Feature Importances als 'RandomForest_Feature_Importances.csv' gespeichert.")
    
    # Auswahl der wichtigsten Features (z.B. Top 10)
    top_features = feature_importances.head(10).index.tolist()
    print("Top Features:", top_features)
    
    # Optional: Speichern der Top-Features für weitere Analysen
    top_features_df = df_normalized[top_features + ['action']]
    top_features_df.to_csv(os.path.join(DATA_DIR, "top_features.csv"), index=False)
    print("Top-Features-Daten als 'top_features.csv' gespeichert.")

def main():
    # Labels der Zustandsmerkmale entsprechend der LunarLander-Umgebung
    labels = [
        "x_position", "y_position", "x_velocity", "y_velocity",
        "angle", "angular_velocity", "left_leg_contact", "right_leg_contact"
    ]
    
    print("Lade das vortrainierte Modell...")
    model = load_model(MODEL_PATH)
    
    print(f"Sammle Daten aus {NUM_EPISODES} Episoden...")
    data = collect_data(model, num_episodes=NUM_EPISODES)
    print(f"Gesammelte {len(data)} Zustands-Aktions-Paare.")
    
    print("Speichere Daten und führe Feature Selection durch...")
    save_data_and_feature_selection(data, labels)
    print(f"Daten und Feature Importances gespeichert in '{DATA_DIR}'.")
    
    # Da Sie nur die Feature-Wichtigkeiten berechnen möchten, benötigen Sie keine weitere Random Forest-Analyse.
    print("Fertig!")

if __name__ == "__main__":
    main()
