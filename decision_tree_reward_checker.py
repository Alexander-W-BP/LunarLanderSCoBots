import gym
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
import joblib
import os
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# ---------------------------
# Evaluation Function
# ---------------------------

def evaluate_tree_policy(tree, scaler, pca, selected_features, env, num_episodes=100, max_steps=1000):
    """
    Bewertet die Performance eines Decision Trees als Policy in der Umgebung.

    Args:
        tree: Der geladene Decision Tree.
        scaler: Der geladene StandardScaler.
        pca: Das geladene PCA-Modell.
        selected_features: Die geladenen ausgewählten Features.
        env: Die Gym-Umgebung.
        num_episodes: Anzahl der Episoden zur Evaluation.
        max_steps: Maximale Schritte pro Episode.

    Returns:
        Mean Reward über alle Episoden.
    """
    total_rewards = []
    
    for episode in range(num_episodes):
        reset_output = env.reset()
        if isinstance(reset_output, tuple):
            state, _ = reset_output  # Entpacken des Tupels (state, info)
        else:
            state = reset_output
        total_reward = 0
        for step in range(max_steps):
            # Datenvorverarbeitung
            state_df = pd.DataFrame([state], columns=['x', 'y', 'vx', 'vy', 'theta', 'v_theta', 'left_leg', 'right_leg'])
            state_selected = state_df[selected_features]
            state_scaled = scaler.transform(state_selected)
            state_pca = pca.transform(state_scaled)
            
            # Aktion vorhersagen
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
            
            if done:
                break
        total_rewards.append(total_reward)
        if (episode + 1) % 10 == 0:
            print(f"Evaluierung: Episode {episode+1}/{num_episodes} abgeschlossen.")
    
    mean_reward = np.mean(total_rewards)
    return mean_reward

# ---------------------------
# Hauptfunktion
# ---------------------------

def main():
    # Pfad zum Verzeichnis, in dem die Modelle und Preprocessing-Artefakte gespeichert sind
    MODEL_DIR = "decision_trees"  # Passen Sie diesen Pfad bei Bedarf an

    # Überprüfen, ob das Modellverzeichnis existiert
    if not os.path.exists(MODEL_DIR):
        print(f"Modellverzeichnis '{MODEL_DIR}' existiert nicht. Stellen Sie sicher, dass die Modelle vorhanden sind.")
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

    # Initialisiere die LunarLander-Umgebung
    env = gym.make('LunarLander-v2')

    # Definieren der Baum-Tiefen, die evaluiert werden sollen
    depths = range(1, 8)

    evaluated_mean_rewards = []

    for depth in depths:
        tree_path = os.path.join(MODEL_DIR, f'decision_tree_depth_{depth}.joblib')
        if not os.path.exists(tree_path):
            print(f"Decision Tree mit Tiefe {depth} nicht gefunden unter '{tree_path}'. Überspringe...")
            evaluated_mean_rewards.append(np.nan)
            continue

        # Laden des Decision Trees
        try:
            tree = joblib.load(tree_path)
            print(f"Decision Tree mit Tiefe {depth} erfolgreich geladen.")
        except Exception as e:
            print(f"Fehler beim Laden des Decision Trees mit Tiefe {depth}: {e}")
            evaluated_mean_rewards.append(np.nan)
            continue

        # Evaluierung des Decision Trees
        print(f"\nEvaluierung von Decision Tree mit Tiefe {depth}...")
        mean_reward = evaluate_tree_policy(tree, scaler, pca, selected_features, env, num_episodes=100, max_steps=1000)
        evaluated_mean_rewards.append(mean_reward)
        print(f"Decision Tree mit Tiefe {depth}: Mean Reward = {mean_reward}")

    env.close()

    # Schritt 8: Plot der Ergebnisse
    print("\nSchritt 8: Plot der Evaluationsergebnisse...")
    plt.figure(figsize=(12, 6))
    plt.plot(depths, evaluated_mean_rewards, marker='o', linestyle='-', color='b')
    plt.title("Mean Reward vs. Decision Tree Depth")
    plt.xlabel("Tree Depth")
    plt.ylabel("Mean Reward")
    plt.grid(True)
    plt.xticks(depths)
    plt.ylim(bottom=min(evaluated_mean_rewards) - 10, top=max(evaluated_mean_rewards) + 10)
    plt.show()

    # Optional: Speichern der Ergebnisse
    results_df = pd.DataFrame({
        'Tree Depth': depths,
        'Mean Reward': evaluated_mean_rewards
    })
    results_csv_path = os.path.join(MODEL_DIR, 'decision_tree_evaluation_results.csv')
    results_df.to_csv(results_csv_path, index=False)
    print(f"Ergebnisse wurden in '{results_csv_path}' gespeichert.")

if __name__ == "__main__":
    main()
