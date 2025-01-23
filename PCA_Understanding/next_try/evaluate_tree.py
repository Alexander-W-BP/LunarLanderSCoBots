#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import joblib
import numpy as np
import pandas as pd
import gym

from stable_baselines3 import PPO
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

###################################
# Die folgenden Imports/Functions
# entsprechen der Logik deines Skripts
###################################

def collect_data(env, agent, num_episodes=100, max_steps=1000):
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
                next_state, reward, done, _ = step_output
            data.append([state, action, reward])
            state = next_state
            if done:
                break
    df = pd.DataFrame(data, columns=["state", "action", "reward"])
    return df

def preprocess_data(df):
    """
    Zerlegt den State-Vektor in einzelne Spalten, entfernt NaNs.
    """
    state_feats = pd.DataFrame(df["state"].tolist(),
                               columns=["x","y","vx","vy","theta","v_theta","left_leg","right_leg"])
    df_out = pd.concat([state_feats, df[["action","reward"]]], axis=1)
    df_out.dropna(inplace=True)
    df_out["action"] = df_out["action"].astype(int)
    return df_out

###################################
# Policy-Evaluation-Funktion 
###################################

def evaluate_tree_policy(tree, scaler, pca, env, selected_features, num_episodes=30, max_steps=1000):
    """
    Misst den Mean Reward eines Decision Trees in LunarLander-v2.
    """
    total_rewards = []
    for ep in range(num_episodes):
        obs = env.reset()
        if isinstance(obs, tuple):
            obs, _info = obs
        ep_reward = 0
        for _ in range(max_steps):
            df_obs = pd.DataFrame([obs],
                                  columns=["x","y","vx","vy","theta","v_theta","left_leg","right_leg"])
            df_obs = df_obs[selected_features]
            obs_scaled = scaler.transform(df_obs)
            obs_pca    = pca.transform(obs_scaled)

            action_pred = tree.predict(obs_pca)[0]
            action_pred = int(action_pred)
            action_pred = np.clip(action_pred, 0, env.action_space.n - 1)

            step_out = env.step(action_pred)
            if len(step_out) == 5:
                next_obs, reward, terminated, truncated, _ = step_out
                done = terminated or truncated
            else:
                next_obs, reward, done, _ = step_out
            ep_reward += reward
            obs = next_obs
            if done:
                break
        total_rewards.append(ep_reward)
    return np.mean(total_rewards)

###################################
# Hauptskript
###################################

def main():
    # -----------------------------
    # 1) Lade oder sammle Daten
    # -----------------------------
    # Option A: Dieselben Daten wie im Training laden (z.B. aus CSV):
    # df_raw = pd.read_csv("...dein_training_dataset.csv")
    #
    # Option B: Wieder PPO laden & neu sammeln
    # (Achtung: Das kann minimal andere Daten sein!)
    env = gym.make("LunarLander-v2")
    ppo_path = "models/ppo_LunarLander-v2/ppo-LunarLander-v2.zip"
    model = PPO.load(ppo_path)
    print("Sammle erneut Daten via PPO-Agent ...")
    df_raw = collect_data(env, model, num_episodes=100, max_steps=1000)
    
    # 2) Preprocess
    df_clean = preprocess_data(df_raw)
    print(f"Gesammelte Datensätze: {df_clean.shape[0]}")
    
    # 3) Lade Preprocessing-Artefakte aus 'decision_trees' 
    #    (damit wir dieselbe Feature-Auswahl, denselben Scaler & PCA haben)
    model_dir = "decision_trees"
    scaler_path = os.path.join(model_dir, "scaler.joblib")
    pca_path = os.path.join(model_dir, "pca.joblib")
    features_path = os.path.join(model_dir, "selected_features.joblib")

    if not (os.path.exists(scaler_path) and os.path.exists(pca_path) and os.path.exists(features_path)):
        print("Fehler: Preprocessing-Dateien nicht gefunden!")
        return

    scaler = joblib.load(scaler_path)
    pca = joblib.load(pca_path)
    selected_features = joblib.load(features_path)
    print("Preprocessing-Artefakte erfolgreich geladen.")

    # 4) Extrahiere X,y für Offline-Accuracy
    #    => X nur die selected_features
    X_all = df_clean[selected_features].values
    y_all = df_clean["action"].values

    # 5) Baue aus X_all -> scaler.transform -> pca.transform
    #    Dafür brauchen wir FITTE Scaler/PCA. 
    #    Achtung: Scaler/PCA wurden im Training fit-transformiert. 
    #    Hier "inverse usage": Wir transformieren nun X_all
    X_all_scaled = scaler.transform(X_all)
    X_all_pca    = pca.transform(X_all_scaled)

    # 6) Train/Test-Split für Offline-Accuracy
    X_train, X_test, y_train, y_test = train_test_split(
        X_all_pca, y_all, test_size=0.2, random_state=42
    )
    print(f"Offline-Accuracy: Train={X_train.shape}, Test={X_test.shape}")

    # 7) Finde alle DecisionTrees im Verzeichnis
    tree_files = [f for f in os.listdir(model_dir)
                  if f.endswith(".joblib") and f.startswith("best_tree_")]

    if not tree_files:
        print("Keine Entscheidungsbäume decision_tree_depth_*.joblib im Ordner gefunden!")
        return

    results = []

    # 8) Evaluate jeden Tree: Offline-Accuracy + Mean Reward
    for tree_file in tree_files:
        tree_path = os.path.join(model_dir, tree_file)
        print(f"\nLade Entscheidungsbaum: {tree_path}")
        tree_model = joblib.load(tree_path)
        
        # 8a) Offline-Accuracy
        y_test_pred = tree_model.predict(X_test)
        acc = accuracy_score(y_test, y_test_pred)

        # 8b) Mean Reward
        mean_rew = evaluate_tree_policy(tree_model, scaler, pca, env,
                                        selected_features,
                                        num_episodes=30, max_steps=1000)

        print(f"{tree_file}: Accuracy={acc:.3f}, MeanReward={mean_rew:.2f}")
        results.append((tree_file, acc, mean_rew))

    env.close()

    # 9) Sortieren & ausgeben
    results.sort(key=lambda x: x[2], reverse=True)  # sortiere nach MeanReward absteigend

    print("\n===== Ergebnisse =====")
    for fname, acc, rew in results:
        print(f"{fname}: Accuracy={acc:.3f}, MeanReward={rew:.2f}")

    best_file, best_acc, best_rew = results[0]
    print(f"\nBester Baum nach Mean Reward: {best_file} (Acc={best_acc:.3f}, MeanReward={best_rew:.2f})")

    # 10) In CSV speichern
    df_out = pd.DataFrame(results, columns=["TreeFile","Accuracy","MeanReward"])
    df_out.to_csv("all_trees_evaluation_results_with_acc.csv", index=False)
    print("Ergebnisse in 'all_trees_evaluation_results_with_acc.csv' gespeichert.")

if __name__ == "__main__":
    main()
