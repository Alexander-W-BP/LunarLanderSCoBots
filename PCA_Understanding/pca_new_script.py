import gym
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
import warnings
import joblib
import os

warnings.filterwarnings("ignore")
from stable_baselines3 import PPO

def collect_data(env, agent, num_episodes, max_steps):
    data = []
    for episode in range(num_episodes):
        reset_output = env.reset()
        state = reset_output[0] if isinstance(reset_output, tuple) else reset_output
        for step in range(max_steps):
            action, _ = agent.predict(state, deterministic=True)
            action = int(action)
            step_output = env.step(action)
            next_state = step_output[0]
            reward = step_output[1]
            done = step_output[2] if len(step_output) < 5 else step_output[2] or step_output[3]
            data.append([state, action, reward])
            state = next_state
            if done:
                break
    return pd.DataFrame(data, columns=['state', 'action', 'reward'])

def evaluate_tree_policy(tree, tree_features, base_features, env, num_episodes=3, max_steps=1000):
    total_rewards = []
    for _ in range(num_episodes):
        state = env.reset()[0]
        episode_reward = 0
        for _ in range(max_steps):
            state_df = pd.DataFrame([state], columns=['x', 'y', 'vx', 'vy', 'theta', 'v_theta', 'left_leg', 'right_leg'])
            state_base = state_df[list(base_features)]
            
            state_data = state_base.copy()
            interaction_pairs = combinations(base_features, 2)
            for feat1, feat2 in interaction_pairs:
                state_data[f'{feat1}*{feat2}'] = state_data[feat1] * state_data[feat2]
            
            try:
                action = tree.predict(state_data[tree_features])[0]
            except:
                action = 0
            
            step_output = env.step(action)
            next_state = step_output[0]
            reward = step_output[1]
            done = step_output[2] if len(step_output) < 5 else step_output[2] or step_output[3]
            
            episode_reward += reward
            state = next_state
            if done:
                break
        total_rewards.append(episode_reward)
    return np.mean(total_rewards)

def analyze_feature_groups(df, env, base_save_dir):
    relevant_features = ['x', 'y', 'vx', 'vy', 'theta', 'v_theta']
    feature_groups = list(combinations(relevant_features, 5))
    
    print(f"\nAnalysiere {len(feature_groups)} 5er-Kombinationen...")

    for group_idx, features in enumerate(feature_groups):
        group_name = "_".join(features)
        group_dir = os.path.join(base_save_dir, group_name)
        os.makedirs(group_dir, exist_ok=True)

        print(f"\n\n—— Gruppe {group_idx+1}/{len(feature_groups)}: {group_name} ——")
        print("==============================================")
        
        all_rewards = []
        pca_equations = []

        for iteration in range(10):
            print(f"\nIteration {iteration+1}/10:")
            iter_dir = os.path.join(group_dir, f"iteration_{iteration+1}")
            os.makedirs(iter_dir, exist_ok=True)

            # Datenvorbereitung
            X_base = df[list(features)]
            y = df['action']

            # PCA mit zufälliger Initialisierung
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_base)
            
            pca = PCA(n_components=5, random_state=iteration)
            X_pca = pca.fit_transform(X_scaled)

            # Decision Tree mit Interaktionstermen
            X_tree = X_base.copy()
            interaction_pairs = combinations(features, 2)
            for feat1, feat2 in interaction_pairs:
                X_tree[f'{feat1}*{feat2}'] = X_tree[feat1] * X_tree[feat2]
            
            tree = DecisionTreeClassifier(max_depth=8, random_state=iteration)
            tree.fit(X_tree, y)
            
            # Evaluation
            mean_reward = evaluate_tree_policy(tree, X_tree.columns, features, env)
            all_rewards.append(mean_reward)

            # Speichern der Modelle
            joblib.dump(tree, os.path.join(iter_dir, 'decision_tree.joblib'))
            joblib.dump(scaler, os.path.join(iter_dir, 'scaler.joblib'))
            joblib.dump(pca, os.path.join(iter_dir, 'pca.joblib'))

            # PCA-Komponenten speichern
            components = pca.components_
            equation_header = f"\nIteration {iteration+1} (Reward: {mean_reward:.2f})"
            pca_equations.append(equation_header)
            
            for i, comp in enumerate(components):
                comp_str = " + ".join([f"{coef:.4f}*{feat}" for coef, feat in zip(comp, features)])
                pca_equations.append(f"PC{i+1}: {comp_str}")

        # Gruppenreport erstellen
        with open(os.path.join(group_dir, 'group_report.txt'), 'w') as f:
            f.write(f"Gruppenanalyse: {group_name}\n")
            f.write(f"Durchschnittlicher Reward: {np.mean(all_rewards):.2f} ± {np.std(all_rewards):.2f}\n")
            f.write("Einzelne Iterationen:\n")
            f.write("\n".join(pca_equations))

def main():
    env = gym.make('LunarLander-v2')
    MODEL_DIR = "C:/Studium_TU_Darmstadt/Master/1. Semester/KI Praktikum/Best_Model/ppo-LunarLander-v2/ppo-LunarLander-v2.zip"
    model = PPO.load(MODEL_DIR)
    
    print("Starte Datensammlung...")
    df = collect_data(env, model, num_episodes=500, max_steps=1000)
    
    state_features = pd.DataFrame(df['state'].tolist(),
                                columns=['x', 'y', 'vx', 'vy', 'theta', 'v_theta', 'left_leg', 'right_leg'])
    df = pd.concat([state_features, df[['action', 'reward']]], axis=1)
    
    base_save_dir = "pca_5feature_10iter_analyses"
    analyze_feature_groups(df, env, base_save_dir)
    
    env.close()
    print("\nAnalyse abgeschlossen! Ergebnisse gespeichert in:", os.path.abspath(base_save_dir))

if __name__ == "__main__":
    main()