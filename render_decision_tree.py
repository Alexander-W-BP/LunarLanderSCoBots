import gym
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier, export_text
import joblib
import os
import warnings
import time

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# ---------------------------
# Evaluation und Visualisierung
# ---------------------------

def preprocess_state(state, scaler, pca, selected_features):
    """
    Wendet die gleichen Vorverarbeitungsschritte an wie beim Training.

    Args:
        state: Der aktuelle Zustand der Umgebung.
        scaler: Der geladene StandardScaler.
        pca: Das geladene PCA-Modell.
        selected_features: Die geladenen ausgewählten Features.

    Returns:
        Transformierter Zustand.
    """
    # Umwandlung des Zustands in einen DataFrame
    state_df = pd.DataFrame([state], columns=['x', 'y', 'vx', 'vy', 'theta', 'v_theta', 'left_leg', 'right_leg'])
    
    # Auswahl der relevanten Features
    state_selected = state_df[selected_features]
    
    # Skalierung der Features
    state_scaled = scaler.transform(state_selected)
    
    # Anwendung von PCA
    state_pca = pca.transform(state_scaled)
    
    return state_pca

def print_decision_tree_text(tree, feature_names):
    """
    Gibt den Entscheidungsbaum als Text aus.

    Args:
        tree: Das geladene DecisionTreeClassifier-Modell.
        feature_names: Die Namen der Features, die im Baum verwendet werden.
    """
    tree_rules = export_text(tree, feature_names=feature_names)
    print("----- Entscheidungsbaum -----")
    print(tree_rules)
    print("-----------------------------\n")

def main():
    # Pfad zum Verzeichnis, in dem die Modelle und Preprocessing-Artefakte gespeichert sind
    MODEL_DIR = "decision_trees"  # Passe diesen Pfad bei Bedarf an

    # Überprüfen, ob das Modellverzeichnis existiert
    if not os.path.exists(MODEL_DIR):
        print(f"Modellverzeichnis '{MODEL_DIR}' existiert nicht. Stelle sicher, dass die Modelle vorhanden sind.")
        return

    # Laden der Preprocessing-Artefakte
    try:
        scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.joblib'))
        pca = joblib.load(os.path.join(MODEL_DIR, 'pca.joblib'))
        selected_features = joblib.load(os.path.join(MODEL_DIR, 'selected_features.joblib'))
        print("Preprocessing-Artefakte erfolgreich geladen.")
    except Exception as e:
        print(f"Fehler beim Laden der Preprocessing-Artefakte: {e}")
        return

    # Laden des Decision Trees mit Tiefe 3
    tree_path = os.path.join(MODEL_DIR, 'decision_tree_depth_3_168.63.joblib')
    if not os.path.exists(tree_path):
        print(f"Decision Tree mit Tiefe 3 nicht gefunden unter '{tree_path}'. Stelle sicher, dass die Datei existiert.")
        return

    try:
        tree = joblib.load(tree_path)
        print("Decision Tree mit Tiefe 3 erfolgreich geladen.")
    except Exception as e:
        print(f"Fehler beim Laden des Decision Trees: {e}")
        return

    # Initialisieren der LunarLander-Umgebung mit Rendering
    env = gym.make('LunarLander-v2', render_mode='human')

    num_episodes = 3  # Anzahl der durchzuführenden Episoden
    max_steps = 1000   # Maximale Schritte pro Episode

    for episode in range(1, num_episodes + 1):
        print(f"\nEpisode {episode}/{num_episodes} startet...")
        
        # Ausgabe des Entscheidungsbaums bei jedem Episodenstart
        print_decision_tree_text(tree, selected_features)
        
        reset_output = env.reset()
        if isinstance(reset_output, tuple):
            state, _ = reset_output  # Entpacken des Tupels (state, info)
        else:
            state = reset_output

        total_reward = 0
        done = False
        step = 0

        while not done and step < max_steps:
            # Preprocessing des Zustands
            state_pca = preprocess_state(state, scaler, pca, selected_features)
            
            # Vorhersage der Aktion
            action_pred = tree.predict(state_pca)[0]
            action = int(action_pred)  # Aktionen sind diskret (0, 1, 2, 3)
            action = np.clip(action, 0, env.action_space.n - 1)  # Sicherstellen, dass Aktion gültig ist

            # Aktion ausführen
            step_output = env.step(action)
            if len(step_output) == 5:
                next_state, reward, terminated, truncated, _ = step_output
                done = terminated or truncated
            else:
                next_state, reward, done, _ = step_output  # Fallback für ältere Gym-Versionen

            total_reward += reward
            state = next_state
            step += 1

            # Kurze Pause, um die Rendering-Geschwindigkeit zu steuern
            time.sleep(0.02)  # 20 ms Pause

        print(f"Episode {episode} abgeschlossen. Gesamte Belohnung: {total_reward}")

    env.close()
    print("\nAlle Episoden abgeschlossen.")

if __name__ == "__main__":
    main()
