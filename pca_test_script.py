import gym
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import joblib
import warnings

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeClassifier, export_text

from stable_baselines3 import PPO

warnings.filterwarnings("ignore")

# ---------------------------
# Schritt 1: Datensammlung
# ---------------------------

def collect_data(env, agent, num_episodes=1000, max_steps=1000):
    data = []
    for episode in range(num_episodes):
        reset_output = env.reset()
        if isinstance(reset_output, tuple):
            state, _ = reset_output  # (state, info)
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
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode+1}/{num_episodes} abgeschlossen.")
    df = pd.DataFrame(data, columns=['state', 'action', 'reward'])
    return df

# ---------------------------
# Schritt 2: Datenaufbereitung
# ---------------------------

def preprocess_data(df):
    state_features = pd.DataFrame(df['state'].tolist(),
                                  columns=['x','y','vx','vy','theta','v_theta','left_leg','right_leg'])
    df = pd.concat([state_features, df[['action','reward']]], axis=1)
    
    df['action'] = df['action'].astype(int)
    
    initial_shape = df.shape
    df.dropna(inplace=True)
    final_shape = df.shape
    print(f"Datenbereinigung: Entfernte {initial_shape[0] - final_shape[0]} Zeilen mit fehlenden Werten.")
    return df

# ---------------------------
# Schritt 3: Feature Selection
# ---------------------------

def feature_selection(df, target='action', top_k=5):
    X = df.drop(['action','reward'], axis=1)
    y = df[target]
    
    unique_labels = y.unique()
    print(f"Einzigartige Labels in '{target}': {unique_labels}")
    
    if not np.issubdtype(y.dtype, np.integer):
        y = y.astype(int)
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    feature_names = X.columns
    
    plt.figure(figsize=(10,6))
    plt.title("Feature Importances")
    plt.bar(range(X.shape[1]), importances[indices], align='center')
    plt.xticks(range(X.shape[1]), feature_names[indices], rotation=45)
    plt.tight_layout()
    plt.show()
    
    selected_features = feature_names[indices[:top_k]]
    print(f"Ausgewählte Features: {list(selected_features)}")
    X_selected = X[selected_features]
    return X_selected, selected_features

# ---------------------------
# Schritt 4: Dimensionsreduktion mit PCA
# ---------------------------

def apply_pca(X, variance_threshold=0.95, random_state=42):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA(n_components=variance_threshold, random_state=random_state)
    X_pca = pca.fit_transform(X_scaled)
    print(f"Anzahl der PCA-Komponenten (Seed={random_state}): {pca.n_components_}")
    
    return X_pca, scaler, pca

# ---------------------------
# Schritt 5: Training der Decision Trees
# ---------------------------

def train_decision_trees(X_train, y_train, depths=range(1,6)):
    trees = []
    for depth in depths:
        tree = DecisionTreeClassifier(max_depth=depth, random_state=42)
        tree.fit(X_train, y_train)
        trees.append(tree)
    return trees

# ---------------------------
# Schritt 6: Evaluation der Decision Trees
# ---------------------------

def evaluate_tree_policy(tree, scaler, pca, env, selected_features,
                         num_episodes=100, max_steps=1000):
    total_rewards = []
    for episode in range(num_episodes):
        reset_output = env.reset()
        if isinstance(reset_output, tuple):
            state, _ = reset_output
        else:
            state = reset_output
        total_reward = 0
        for step in range(max_steps):
            state_df = pd.DataFrame([state], 
                                    columns=['x','y','vx','vy','theta','v_theta','left_leg','right_leg'])
            state_selected = state_df[selected_features]
            state_scaled = scaler.transform(state_selected)
            state_pca = pca.transform(state_scaled)
            
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
    return np.mean(total_rewards)

# ---------------------------
# Hauptfunktion mit Loop über Seeds
# ---------------------------

def main():
    # 0) Environment & PPO laden
    env = gym.make('LunarLander-v2')
    MODEL_DIR = "models/ppo_LunarLander-v2/ppo-LunarLander-v2.zip"
    print("Lade PPO-Modell...")
    try:
        model = PPO.load(MODEL_DIR)
        print("PPO-Modell erfolgreich geladen.")
    except Exception as e:
        print(f"Fehler beim Laden des PPO-Modells: {e}")
        return
    
    # 1) Einmal Daten sammeln
    print("\n[1/3] Datensammlung...")
    df = collect_data(env, model, num_episodes=100, max_steps=1000)
    df = preprocess_data(df)
    
    # 2) Feature Selection (einmalig)
    print("\n[2/3] Feature Selection...")
    X_selected, selected_features = feature_selection(df, target='action', top_k=5)
    y = df['action']
    
    # 3) Schleife über verschiedene PCA-Seeds
    pca_seeds = [0,1,2,3,4,5,6,7,8,9]
    results_all = []  # Speichert (seed, depth, mean_reward)

    for seed in pca_seeds:
        print(f"\n--- PCA-Analyse mit Seed={seed} ---")
        
        # PCA
        X_pca, scaler, pca_model = apply_pca(X_selected, variance_threshold=0.95, 
                                             random_state=seed)
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X_pca, y, test_size=0.2, random_state=42
        )
        
        # Decision Trees trainieren
        depths = range(1,6)
        trees = train_decision_trees(X_train, y_train, depths=depths)
        
        # Evaluation
        for idx, depth in enumerate(depths):
            tree = trees[idx]
            mean_reward = evaluate_tree_policy(
                tree, scaler, pca_model, env, selected_features,
                num_episodes=50,  # ggf. anpassen, um Zeit zu sparen
                max_steps=1000
            )
            print(f"  -> Depth={depth}, Mean Reward={mean_reward:.2f}")
            results_all.append([seed, depth, mean_reward])
    
    env.close()
    
    # Ergebnisse in DataFrame packen und speichern
    results_df = pd.DataFrame(results_all, columns=["PCA_Seed","Tree_Depth","Mean_Reward"])
    results_df.to_csv("decision_tree_evaluation_multiseed.csv", index=False)
    print("\nAlle Ergebnisse wurden in 'decision_tree_evaluation_multiseed.csv' gespeichert.")
    
    # Plotten: Rewards pro Seed und Tiefe
    plt.figure(figsize=(10,6))
    for seed in pca_seeds:
        subset = results_df[results_df["PCA_Seed"] == seed]
        plt.plot(subset["Tree_Depth"], subset["Mean_Reward"], marker='o', label=f"Seed={seed}")
    plt.title("Mean Reward vs. Decision Tree Depth (verschiedene PCA-Seeds)")
    plt.xlabel("Tree Depth")
    plt.ylabel("Mean Reward")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
