#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gym
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
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
    data = []
    for episode in range(num_episodes):
        reset_output = env.reset()
        if isinstance(reset_output, tuple):
            state, _info = reset_output
        else:
            state = reset_output
        for step in range(max_steps):
            action, _ = agent.predict(state, deterministic=True)
            action = int(action)
            step_output = env.step(action)
            if len(step_output) == 5:
                next_state, reward, terminated, truncated, _ = step_output
                done = terminated or truncated
            else:
                next_state, reward, done, _info = step_output
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
    """
    state_features = pd.DataFrame(df['state'].tolist(),
                                  columns=['x', 'y', 'vx', 'vy', 'theta', 'v_theta', 'left_leg', 'right_leg'])
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
    Wählt die wichtigsten Features basierend auf der Feature-Wichtigkeit eines RandomForestClassifiers.
    """
    X = df.drop(['action', 'reward'], axis=1)
    y = df[target]
    
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

def apply_pca(X, variance_threshold=0.99):
    """
    Wendet PCA zur Dimensionsreduktion an und behält den angegebenen Varianzanteil.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA(n_components=variance_threshold, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    
    print(f"Anzahl der PCA-Komponenten (Var >= {variance_threshold}): {pca.n_components_}")
    return X_pca, scaler, pca

# ---------------------------
# Schritt 5: Training der Decision Trees
# ---------------------------

def train_decision_trees(X_train, y_train, X_test, y_test, depth):
    """
    Trainiert mehrere Decision Trees (Tiefe=depth) mit unterschiedlichen
    Hyperparametern und wählt den besten basierend auf Test-Accuracy aus.

    Returns: (bester_tree, alle_trees, best_accuracy)
    """
    # Verschiedene Parameter-Kombinationen, die wir einfach durchprobieren
    # (ohne GridSearchCV).
    criterions = ["gini", "entropy"]
    min_splits = [2, 5, 10]
    min_leafs = [1, 2, 5]

    best_acc = -1.0
    best_tree = None
    all_trees = []
    
    for criterion in criterions:
        for ms_split in min_splits:
            for ms_leaf in min_leafs:
                # random_state wird variiert, um "verschiedene" Bäume zu bekommen
                # (du kannst hier z. B. range(5) loops machen, etc.)
                for seed in range(3):  
                    dt = DecisionTreeClassifier(
                        max_depth=depth,
                        criterion=criterion,
                        min_samples_split=ms_split,
                        min_samples_leaf=ms_leaf,
                        random_state=seed
                    )
                    dt.fit(X_train, y_train)

                    # Test-Accuracy
                    y_pred_test = dt.predict(X_test)
                    acc_test = accuracy_score(y_test, y_pred_test)

                    all_trees.append(dt)

                    if acc_test > best_acc:
                        best_acc = acc_test
                        best_tree = dt

    print(f"\nBester Tree offline-Accuracy: {best_acc:.4f}")
    return best_tree, all_trees, best_acc

# ---------------------------
# Schritt 6: Evaluation
# ---------------------------

def evaluate_tree_policy(tree, scaler, pca, env, selected_features, num_episodes, max_steps):
    """
    Bewertet die Performance eines Decision Trees als Policy in der Umgebung.
    """
    total_rewards = []
    
    for episode in range(num_episodes):
        reset_output = env.reset()
        if isinstance(reset_output, tuple):
            state, _ = reset_output
        else:
            state = reset_output
        total_reward = 0
        for step in range(max_steps):
            state_df = pd.DataFrame([state], columns=['x', 'y', 'vx', 'vy', 'theta', 'v_theta', 'left_leg', 'right_leg'])
            state_selected = state_df[selected_features]
            state_scaled = scaler.transform(state_selected)
            state_pca = pca.transform(state_scaled)
            
            # Aktion vorhersagen
            action_pred = tree.predict(state_pca)[0]
            action = int(action_pred)
            action = np.clip(action, 0, env.action_space.n - 1)
            
            step_output = env.step(action)
            if len(step_output) == 5:
                next_state, reward, terminated, truncated, _ = step_output
                done = terminated or truncated
            else:
                next_state, reward, done, _ = step_output
            total_reward += reward
            state = next_state
            
            if done:
                break
        total_rewards.append(total_reward)
    
    mean_reward = np.mean(total_rewards)
    return mean_reward

# ---------------------------
# Schritt 7: Hauptfunktion
# ---------------------------

def main():
    # Initialisiere die LunarLander-Umgebung
    env = gym.make('LunarLander-v2')
    
    # Schritt 1: Laden des PPO-Modells
    MODEL_DIR = "models/ppo_LunarLander-v2/ppo-LunarLander-v2.zip"
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
    
    # Schritt 4: Feature Selection (=> Hier top_k=8 statt 5!)
    print("\nSchritt 4: Feature Selection...")
    X_selected, selected_features = feature_selection(df, top_k=8, target='action')
    
    # Schritt 5: Dimensionsreduktion mit PCA (=> var_threshold=0.99)
    print("\nSchritt 5: Dimensionsreduktion mit PCA...")
    X_pca, scaler, pca = apply_pca(X_selected, variance_threshold=0.99)
    
    # Schritt 6: Training => wir trainieren mehrere Bäume, wählen den besten anhand OFFLINE-Accuracy
    print("\nSchritt 6: Training der Decision Trees...")
    X = X_pca
    y = df['action']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Trainingsdaten: {X_train.shape[0]} Zeilen, Testdaten: {X_test.shape[0]} Zeilen.")
    
    # Trainiere + wähle den besten
    best_tree, all_trees, best_acc = train_decision_trees(X_train, y_train, X_test, y_test, depth=3)
    print(f"\n=> Bester Baum (Tiefe=3) mit Offline-Accuracy: {best_acc:.4f}")
    
    # Schritt 7: Evaluate in environment
    print("\nSchritt 7: Evaluation im LunarLander-Umfeld...")
    mean_reward = evaluate_tree_policy(best_tree, scaler, pca, env, selected_features, num_episodes=30, max_steps=1000)
    print(f"Bester Baum: Mean Reward = {mean_reward:.2f}")

    # Speichern
    MODEL_SAVE_DIR = "decision_trees_5"
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

    # Baum als Text
    pca_feature_names = [f'PC{i}' for i in range(1, pca.n_components_ + 1)]
    tree_text = export_text(best_tree, feature_names=pca_feature_names)
    with open(os.path.join(MODEL_SAVE_DIR, 'best_decision_tree.txt'), 'w') as f:
        f.write(tree_text)
    print(f"\nEntscheidungsbaum in '{MODEL_SAVE_DIR}/best_decision_tree.txt' gespeichert.")

    joblib.dump(best_tree, os.path.join(MODEL_SAVE_DIR, f'best_tree_depth5.joblib'))
    print(f"Bester Tree als Joblib-Datei gespeichert.")
    
    # Preprocessing
    joblib.dump(scaler, os.path.join(MODEL_SAVE_DIR, 'scaler.joblib'))
    joblib.dump(pca, os.path.join(MODEL_SAVE_DIR, 'pca.joblib'))
    joblib.dump(selected_features, os.path.join(MODEL_SAVE_DIR, 'selected_features.joblib'))
    print("Preprocessing-Artefakte (Scaler, PCA, Features) gespeichert.")

    # Optional: Mean Rewards pro Tree?
    # (wir könnten alle Bäume bewerten, aber wir haben bereits den besten basierend auf Offline-Accuracy.)
    
    env.close()
    print("\nFertig!")

if __name__ == "__main__":
    main()
