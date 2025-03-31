import gym
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
import warnings
import joblib
import os

# Warnings unterdrücken für Übersichtlichkeit
warnings.filterwarnings("ignore")

# OPTIONAL: Wenn du den PPO-Agenten nutzt
from stable_baselines3 import PPO

# ---------------------------------
# 1) Datensammlung (OPTIONAL)
# ---------------------------------
def collect_data(env, agent, num_episodes=50, max_steps=1000):
    """
    Sammelt Daten mithilfe eines (bereits trainierten) PPO-Agenten.
    Gibt einen DataFrame mit Spalten ['state', 'action', 'reward'] zurück.
    """
    data = []
    for episode in range(num_episodes):
        reset_output = env.reset()
        if isinstance(reset_output, tuple):
            state, _ = reset_output  # neuere Gym-Version
        else:
            state = reset_output     # ältere Gym-Version

        for step in range(max_steps):
            action, _ = agent.predict(state, deterministic=True)
            action = int(action)

            step_output = env.step(action)
            if len(step_output) == 5:
                next_state, reward, terminated, truncated, _ = step_output
                done = terminated or truncated
            else:
                next_state, reward, done, _ = step_output

            data.append([state, action, reward])
            state = next_state
            if done:
                break

    df = pd.DataFrame(data, columns=['state', 'action', 'reward'])
    return df


# ---------------------------------
# 2) Datenaufbereitung
# ---------------------------------
def preprocess_data(df):
    """
    Erwartet einen DataFrame mit Spalte 'state' (Zustandsvektor),
    erstellt separate Spalten für jeden Zustandswert,
    konvertiert 'action' zu int, entfernt NaNs.
    """
    # State in einzelne Features zerlegen
    state_features = pd.DataFrame(df['state'].tolist(), 
                                  columns=['x', 'y', 'vx', 'vy', 
                                           'theta', 'v_theta', 
                                           'left_leg', 'right_leg'])
    df = pd.concat([state_features, df[['action', 'reward']]], axis=1)
    
    # Aktion sicherheitshalber in int konvertieren
    df['action'] = df['action'].astype(int)
    
    # NaNs entfernen
    init_shape = df.shape
    df.dropna(inplace=True)
    final_shape = df.shape
    print(f"Datenbereinigung: Entfernte {init_shape[0] - final_shape[0]} Zeilen mit fehlenden Werten.")
    return df

# ---------------------------------
# 3) Training der Decision Trees
# ---------------------------------
def train_decision_trees(X_train, y_train, depth, num_trees=3):
    """
    Trainiert mehrere Bäume derselben Tiefe (default=3 Bäume) 
    und gibt sie als Liste zurück.
    """
    trees = []
    for i in range(num_trees):
        tree = DecisionTreeClassifier(max_depth=depth, random_state=i)
        tree.fit(X_train, y_train)
        trees.append(tree)
    return trees

# ---------------------------------
# 4) (Optionale) Evaluation im Env
# ---------------------------------
def evaluate_tree_policy(tree, env, scaler, pca, selected_features, num_episodes=30, max_steps=1000):
    """
    Bewertet die Performance (Mean Reward) eines Decision Trees als Policy 
    direkt in der angegebenen Gym-Umgebung.
    """
    rewards_all_episodes = []

    for _ in range(num_episodes):
        reset_output = env.reset()
        if isinstance(reset_output, tuple):
            state, _ = reset_output
        else:
            state = reset_output

        episode_reward = 0
        for _ in range(max_steps):
            # DataFrame mit dem aktuellen Zustand
            state_df = pd.DataFrame([state], 
                                    columns=['x', 'y', 'vx', 'vy', 
                                             'theta', 'v_theta', 
                                             'left_leg', 'right_leg'])
            # Nur die gewählten Features nehmen
            state_selected = state_df[selected_features]
            # Skalieren und PCA transformieren
            state_scaled = scaler.transform(state_selected)
            state_pca = pca.transform(state_scaled)

            # Aktion vom Tree
            action_pred = tree.predict(state_pca)[0]
            action = int(action_pred)
            action = np.clip(action, 0, env.action_space.n - 1)

            # Schritt ausführen
            step_output = env.step(action)
            if len(step_output) == 5:
                next_state, reward, terminated, truncated, _ = step_output
                done = terminated or truncated
            else:
                next_state, reward, done, _ = step_output

            episode_reward += reward
            state = next_state
            if done:
                break

        rewards_all_episodes.append(episode_reward)

    mean_reward = np.mean(rewards_all_episodes)
    return mean_reward

# ---------------------------------
# Hauptskript
# ---------------------------------
def main():
    # Pfad für die bereits vorhandenen Artefakte (scaler, pca, features)
    BEST_DIR = "decision_trees_best"

    # 1) Lade PCA, Scaler und Selected Features
    scaler_path = os.path.join(BEST_DIR, "scaler.joblib")
    pca_path = os.path.join(BEST_DIR, "pca.joblib")
    features_path = os.path.join(BEST_DIR, "selected_features.joblib")

    scaler = joblib.load(scaler_path)
    pca = joblib.load(pca_path)
    selected_features = joblib.load(features_path)

    print("Geladene Artefakte:")
    print("  - Scaler:", scaler)
    print("  - PCA:", pca)
    print("  - Selected Features:", selected_features)

    # 2) Umgebung & PPO-Agent laden ODER eigene Datenquelle nutzen
    env = gym.make('LunarLander-v2')

    # PPO-Agent laden (wenn du einen Agenten hast)
    # Pfad zu deinem PPO-Modell anpassen
    ppo_model_path = "models\\ppo_LunarLander-v2\\ppo-LunarLander-v2.zip"
    try:
        model = PPO.load(ppo_model_path)
        print("PPO-Modell erfolgreich geladen.")
    except Exception as e:
        print(f"Fehler beim Laden des PPO-Modells: {e}")
        model = None

    # 3) Datensammlung (falls du eine neue Datensammlung machen willst)
    if model is not None:
        print("\nSammle Daten mit dem PPO-Agenten...")
        df = collect_data(env, model, num_episodes=50, max_steps=1000)
    else:
        print("Kein PPO-Modell vorhanden, bitte Daten anderweitig laden.")
        return

    print(f"Größe des ursprünglichen DataFrames: {df.shape}")

    # 4) Datenaufbereitung
    df = preprocess_data(df)
    print(f"Größe nach Preprocessing: {df.shape}")

    # 5) Features für Training vorbereiten
    #    Die Features haben wir schon in 'selected_features'
    X = df[selected_features]
    y = df['action']  # Wir trainieren Bäume, um 'action' zu klassifizieren

    # Jetzt transformieren wir X mit dem geladenen Scaler und PCA
    X_scaled = scaler.transform(X)
    X_pca = pca.transform(X_scaled)

    # Split in Train/Test
    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, 
                                                        test_size=0.2, 
                                                        random_state=1)
    print(f"Trainingsdaten: {X_train.shape}, Testdaten: {X_test.shape}")

    # 6) Training der Bäume (Tiefe 1 bis Tiefe 6)
    NEW_TREES_DIR = "decision_tree_new"
    os.makedirs(NEW_TREES_DIR, exist_ok=True)

    # Liste für den "besten" Reward pro Tiefe
    best_rewards_per_depth = []

    # Wir trainieren jetzt Bäume von Tiefe 1 bis 6 (jeweils 3 Bäume) 
    # und speichern den jeweils besten Reward in best_rewards_per_depth
    for depth in range(1, 7):
        print(f"\n--- Decision Trees mit Tiefe = {depth} ---")
        trees = train_decision_trees(X_train, y_train, depth=depth, num_trees=3)

        best_reward_for_depth = float("-inf")
        best_tree_idx = -1

        # Speichere alle Bäume (Joblib + txt), evaluiere Rewards
        for idx, tree in enumerate(trees):
            # Decision Tree als Joblib
            tree_filename = f"decision_tree_depth_{depth}_tree_{idx}.joblib"
            joblib.dump(tree, os.path.join(NEW_TREES_DIR, tree_filename))

            # Optional: Baumstruktur als Textdatei
            pca_feature_names = [f"PC{i}" for i in range(1, pca.n_components_ + 1)]
            tree_as_text = export_text(tree, feature_names=pca_feature_names)
            txt_filename = f"tree_depth_{depth}_idx_{idx}.txt"
            with open(os.path.join(NEW_TREES_DIR, txt_filename), "w") as f:
                f.write(tree_as_text)

            # Evaluate diesen konkreten Tree im Env
            mean_reward = evaluate_tree_policy(tree, env, scaler, pca, 
                                               selected_features, 
                                               num_episodes=30, 
                                               max_steps=1000)
            print(f"  > Tree {idx}: Reward = {mean_reward:.2f}")

            # Ist dieser Baum besser als alle bisherigen in dieser Tiefe?
            if mean_reward > best_reward_for_depth:
                best_reward_for_depth = mean_reward
                best_tree_idx = idx

        # Ausgabe zum besten Baum
        print(f"  >> Bester Baum für Tiefe {depth} ist Tree {best_tree_idx} mit Reward = {best_reward_for_depth:.2f}")
        best_rewards_per_depth.append(best_reward_for_depth)

    # 7) Ergebnisse in CSV oder Plot
    results_df = pd.DataFrame({
        'Depth': range(1, 7),
        'Best Reward': best_rewards_per_depth
    })
    results_csv_path = os.path.join(NEW_TREES_DIR, "results_depth_1_6.csv")
    results_df.to_csv(results_csv_path, index=False)
    print(f"\nErgebnisse wurden in {results_csv_path} gespeichert.")

    # Plot der besten Rewards pro Tiefe
    plt.figure()
    plt.plot(range(1, 7), best_rewards_per_depth)
    plt.xlabel("Tiefe des Baumes")
    plt.ylabel("Bester Mean Reward")
    plt.title("Decision Tree Tiefe vs. (Bester) Mean Reward")
    plt.show()

    env.close()


if __name__ == "__main__":
    main()
