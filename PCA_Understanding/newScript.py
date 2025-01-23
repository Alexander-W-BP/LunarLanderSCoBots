import gym
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeClassifier, export_text  # Standard Decision Tree
import warnings
import joblib
import os

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Import PPO from stable_baselines3
from stable_baselines3 import PPO

# ---------------------------
# Schritt 1: Datensammlung
# ---------------------------

def collect_data(env, agent, num_episodes, max_steps):
    """
    Sammelt Daten von der Interaktion des PPO-Agenten mit der Umgebung.

    Args:
        env: Die Gym-Umgebung.
        agent: Der PPO-Agent.
        num_episodes: Anzahl der Episoden zur Datensammlung.
        max_steps: Maximale Schritte pro Episode.

    Returns:
        Pandas DataFrame mit den gesammelten Daten.
    """
    data = []
    for episode in range(num_episodes):
        reset_output = env.reset()
        if isinstance(reset_output, tuple):
            state, _ = reset_output  # Entpacken des Tupels (state, info)
        else:
            state = reset_output
        for step in range(max_steps):
            action, _ = agent.predict(state, deterministic=True)  # PPO-Modell zur Aktionsvorhersage
            action = int(action)  # Sicherstellen, dass Aktion ein Integer ist
            step_output = env.step(action)
            if len(step_output) == 5:
                next_state, reward, terminated, truncated, _ = step_output
                done = terminated or truncated
            else:
                next_state, reward, done, _ = step_output  # Fallback für ältere Gym-Versionen
            data.append([state, action, reward])
            state = next_state
            if done:
                break
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode+1}/{num_episodes} abgeschlossen.")
    df = pd.DataFrame(data, columns=['state', 'action', 'reward'])
    return df

# ---------------------------
# Schritt 2: Datenaufbereitung
# ---------------------------

def preprocess_data(df):
    """
    Zerlegt die Zustandsvektoren in einzelne Features und bereinigt die Daten.

    Args:
        df: Der ursprüngliche DataFrame mit gesammelten Daten.

    Returns:
        Bereinigter DataFrame mit getrennten Features.
    """
    # Zerlege den State-Vektor in einzelne Features
    state_features = pd.DataFrame(df['state'].tolist(), columns=['x', 'y', 'vx', 'vy', 'theta', 'v_theta', 'left_leg', 'right_leg'])
    df = pd.concat([state_features, df[['action', 'reward']]], axis=1)
    
    # Sicherstellen, dass 'action' als Integer formatiert ist
    df['action'] = df['action'].astype(int)
    
    # Datenbereinigung
    initial_shape = df.shape
    df.dropna(inplace=True)
    final_shape = df.shape
    print(f"Datenbereinigung: Entfernte {initial_shape[0] - final_shape[0]} Zeilen mit fehlenden Werten.")
    return df

# ---------------------------
# Schritt 3: Feature Selection
# ---------------------------

def feature_selection(df, top_k, target='action'):
    """
    Wählt die wichtigsten Features basierend auf der Feature-Wichtigkeit eines Random Forest Classifiers aus.

    Args:
        df: Der bereinigte DataFrame.
        target: Die Zielvariable ('action' oder 'reward').
        top_k: Anzahl der zu wählenden Top-Features.

    Returns:
        Tuple aus den ausgewählten Features und deren Namen.
    """
    X = df.drop(['action', 'reward'], axis=1)
    y = df[target]
    
    # Überprüfen der einzigartigen Werte in y
    unique_labels = y.unique()
    print(f"Einzigartige Labels in '{target}': {unique_labels}")
    
    # Sicherstellen, dass y diskrete Klassen enthält
    if not np.issubdtype(y.dtype, np.integer):
        print(f"Warnung: Die Zielvariable '{target}' ist nicht vom Typ Integer.")
        y = y.astype(int)
        print(f"Zielvariable '{target}' wurde in Integer konvertiert.")
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    feature_names = X.columns
    
    # Auswahl der Top-k Features
    selected_features = feature_names[indices[:top_k]]
    print(f"Ausgewählte Features: {list(selected_features)}")
    X_selected = X[selected_features]
    return X_selected, selected_features

# ---------------------------
# Schritt 4: Dimensionsreduktion mit PCA
# ---------------------------

def apply_pca(X, variance_threshold=0.95):
    """
    Wendet PCA zur Dimensionsreduktion an und behält einen bestimmten Varianzanteil.

    Args:
        X: Die ausgewählten Features.
        variance_threshold: Anteil der beibehaltenen Varianz.

    Returns:
        Tuple aus den PCA-transformierten Daten, dem Skalierer und dem PCA-Modell.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA(n_components=variance_threshold, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    
    print(f"Anzahl der PCA-Komponenten: {pca.n_components_}")
    return X_pca, scaler, pca

# ---------------------------
# Schritt 5: Training der Decision Trees
# ---------------------------

def train_decision_trees(X_train, y_train, depth, num_trees):
    """
    Trainiert Decision Trees mit unterschiedlichen Tiefen.

    Args:
        X_train: Trainingsdaten.
        y_train: Trainingslabels.
        depths: Bereich der Baumtiefen.

    Returns:
        Liste der trainierten Bäume.
    """
    trees = []
    
    for i in range(num_trees):
        tree = DecisionTreeClassifier(max_depth=depth, random_state=i)
        tree.fit(X_train, y_train)
        trees.append(tree)
        print(f"Trainiert Decision Tree mit Tiefe {depth}.")
    return trees

# ---------------------------
# Schritt 6: Evaluation der Decision Trees
# ---------------------------

def evaluate_tree_policy(tree, scaler, pca, env, selected_features, num_episodes, max_steps):
    """
    Bewertet die Performance eines Decision Trees als Policy in der Umgebung.

    Args:
        tree: Der trainierte Decision Tree.
        scaler: Der Skalierer, der auf die Features angewendet wurde.
        pca: Das PCA-Modell, das auf die Features angewendet wurde.
        env: Die Gym-Umgebung.
        selected_features: Die ausgewählten Features.
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
# Schritt 7: Hauptfunktion
# ---------------------------

def main():
    # Initialisiere die LunarLander-Umgebung
    env = gym.make('LunarLander-v2')
    
    # Schritt 1: Laden des PPO-Modells
    MODEL_DIR = "C:/Studium_TU_Darmstadt/Master/1. Semester/KI Praktikum/Best_Model/ppo-LunarLander-v2/ppo-LunarLander-v2.zip"
    print("Schritt 1: Laden des PPO-Modells...")
    try:
        model = PPO.load(MODEL_DIR)
        print("PPO-Modell erfolgreich geladen.")
    except Exception as e:
        print(f"Fehler beim Laden des PPO-Modells: {e}")
        return
    
    # Schritt 2: Datensammlung
    print("\nSchritt 2: Datensammlung...")
    df = collect_data(env, model, num_episodes=100, max_steps=1000)
    print(f"Gesammelte Daten: {df.shape[0]} Zeilen und {df.shape[1]} Spalten.")
    



    # Schritt 3: Datenaufbereitung
    print("\nSchritt 3: Datenaufbereitung...")
    df = preprocess_data(df)
    print(f"Bereinigte Daten: {df.shape[0]} Zeilen und {df.shape[1]} Spalten.")
    




    # Schritt 4: Feature Selection (Random Forest)
    print("\nSchritt 4: Feature Selection...")
    X_selected, selected_features = feature_selection(df, top_k=5, target='action')
    
    # Schritt 5: Dimensionsreduktion mit PCA
    print("\nSchritt 5: Dimensionsreduktion mit PCA...")
    X_pca, scaler, pca = apply_pca(X_selected, variance_threshold=0.95)




    
    # Schritt 6: Training der Decision Trees
    print("\nSchritt 6: Training der Decision Trees...")
    X = X_pca
    y = df['action']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Trainingsdaten: {X_train.shape[0]} Zeilen, Testdaten: {X_test.shape[0]} Zeilen.")
    
    trees = train_decision_trees(X_train, y_train, depth=3, num_trees=10)
    
    
    # Schritt 7: Evaluation der Decision Trees
    print("\nSchritt 7: Evaluation der Decision Trees im LunarLander-Umfeld...")
    evaluated_mean_rewards = []

    MODEL_SAVE_DIR = "decision_trees"
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

    for idx, tree in enumerate(trees):
        print(f"\nEvaluierung von Decision Tree {idx}...")
        mean_reward = evaluate_tree_policy(tree, scaler, pca, env, selected_features, num_episodes=30, max_steps=1000)
        evaluated_mean_rewards.append(mean_reward)
        print(f"Decision Tree {idx}: Mean Reward = {mean_reward}")

        # --- Speichern des Entscheidungsbaums als Textdatei ---
        # Generieren der PCA-Komponentennamen
        pca_feature_names = [f'PC{i}' for i in range(1, pca.n_components_ + 1)]
        tree_text = export_text(tree, feature_names=pca_feature_names)
        with open(f'decision_tree_depth_{idx}.txt', 'w') as f:
            f.write(tree_text)
        print(f"Entscheidungsbaum {idx} in 'decision_tree_depth_{idx}.txt' gespeichert.")

        # --- Speichern des Entscheidungsbaums als Joblib-Datei ---
        joblib.dump(tree, os.path.join(MODEL_SAVE_DIR, f'decision_tree_depth_{idx}.joblib'))
        print(f"Entscheidungsbaum {idx} als Joblib-Datei gespeichert.")

    # --- Speichern der Preprocessing-Artefakte ---
    joblib.dump(scaler, os.path.join(MODEL_SAVE_DIR, 'scaler.joblib'))
    joblib.dump(pca, os.path.join(MODEL_SAVE_DIR, 'pca.joblib'))
    joblib.dump(selected_features, os.path.join(MODEL_SAVE_DIR, 'selected_features.joblib'))
    print("Preprocessing-Artefakte (Scaler, PCA, ausgewählte Features) gespeichert.")
    env.close()

    
    # Optional: Speichern der Ergebnisse
    # Speichern Sie die Mean Rewards in einer CSV-Datei
    results_df = pd.DataFrame({
        'Mean Reward': evaluated_mean_rewards
    })
    results_df.to_csv('decision_tree_evaluation_results.csv', index=False)
    print("Ergebnisse wurden in 'decision_tree_evaluation_results.csv' gespeichert.")
    
    # Optional: Speichern eines spezifischen Decision Trees
    # Beispiel: Speichern des Baums mit der besten Leistung
    best_index = np.argmax(evaluated_mean_rewards)
    best_tree = trees[best_index]

if __name__ == "__main__":
    main()
