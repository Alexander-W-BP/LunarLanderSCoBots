# train_original_features.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
import gymnasium as gym
from stable_baselines3 import PPO
import os
import re
import pandas
import joblib
from tqdm import tqdm  # Importiere tqdm für Fortschrittsbalken

def load_model(model_path):
    return PPO.load(model_path)

def transform_obs_4meta(obs):
    x_space, y_space, vel_x_space, vel_y_space, angle, angular_vel, leg_1, leg_2 = obs
    # Originale Features
    return np.array([
        x_space, y_space, vel_x_space, vel_y_space,
        angle, angular_vel, leg_1, leg_2
    ], dtype=np.float32)

def evaluate_tree(env, clf, transform_func=None, n_episodes=50, max_steps=1000):
    """
    Führt n_episodes lang den Decision Tree in env aus.
    Gibt (mean_reward, std_reward) zurück.
    """
    rewards = []
    for ep in tqdm(range(n_episodes), desc="Evaluating Episodes", leave=False):
        obs = env.reset(seed=None)[0]
        total_r = 0.0
        for _ in range(max_steps):
            if transform_func:
                obs_transformed = transform_func(obs)
                action = clf.predict(obs_transformed.reshape(1, -1))[0]
            else:
                action = clf.predict(obs.reshape(1, -1))[0]

            obs, reward, done, _, _ = env.step(action)
            total_r += reward
            if done:
                break
        rewards.append(total_r)

    return np.mean(rewards), np.std(rewards)

def gather_performance(model_path, env_name, transform_func=None,
                       num_samples=10000, n_episodes=50, seeds=list(range(6))):
    """
    - Sammelt num_samples Daten mithilfe des PPO-Modells.
    - Trainiert Decision Trees (max_depth=1..15).
    - Für jedes Modell und jeden Seed wird evaluate_tree(...) aufgerufen (n_episodes pro Seed).
    - Mittelt über alle Seeds => finaler mean Reward + std.
    """
    from sklearn.tree import DecisionTreeClassifier

    MEAN_REWARD_THRESHOLD = 100

    # ---- Daten sammeln mit PPO ----
    env = gym.make(env_name)
    model = load_model(model_path)
    obs_list, act_list = [], []
    obs = env.reset(seed=None)[0]
    for _ in tqdm(range(num_samples), desc="Collecting Data", leave=False):
        action, _ = model.predict(obs, deterministic=True)
        obs_list.append(obs)
        act_list.append(action)
        obs, _, done, _, _ = env.step(action)
        if done:
            obs = env.reset(seed=None)[0]

    obs_list = np.array(obs_list)
    act_list = np.array(act_list)

    if transform_func:
        obs_list = np.array([transform_func(o) for o in obs_list])

    # -> Hier kein Split, da wir nur Reward messen (oder optional 100% train)
    depths = range(1, 16)
    mean_rewards = []
    std_rewards = []
    best_tree = None
    best_tree_depth = -1
    trees_above_threshold_with_depths = []

    eval_env = gym.make(env_name)

    for depth in tqdm(depths, desc="Training Trees"):
        clf = DecisionTreeClassifier(max_depth=depth, random_state=0)
        clf.fit(obs_list, act_list)

        # Mehrere Seeds -> Mittelwert
        all_seeds_rewards = []
        for s in tqdm(seeds, desc=f"Evaluating Depth {depth}", leave=False):
            eval_env.reset(seed=s)  # setze seed
            mr, _ = evaluate_tree(eval_env, clf, transform_func=transform_func,
                                 n_episodes=n_episodes, max_steps=1000)
            all_seeds_rewards.append(mr)

        # Mittelwert und Streuung über seeds
        all_seeds_rewards = np.array(all_seeds_rewards)
        current_mean_reward = all_seeds_rewards.mean()
        mean_rewards.append(current_mean_reward)
        std_rewards.append(all_seeds_rewards.std())

        # Überprüfe, ob der aktuelle Baum der bisher beste ist
        if current_mean_reward > MEAN_REWARD_THRESHOLD and (best_tree is None or depth < best_tree_depth):
            best_tree = clf
            best_tree_depth = depth
        
        # Alle Bäume mit mean reward über Threshold speichern
        if current_mean_reward > MEAN_REWARD_THRESHOLD:
            trees_above_threshold_with_depths.append((clf, depth))

    return depths, mean_rewards, std_rewards, best_tree, trees_above_threshold_with_depths

def main():
    MODEL_PATH = "models/ppo_LunarLander-v2/ppo-LunarLander-v2.zip"  # Passe den Pfad an!
    ENV_NAME = "LunarLander-v2"
    OUTPUT_DIR = "decision_tree_models_original"

    # Stelle sicher, dass das Ausgabeverzeichnis existiert
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ---- Originale Features
    depths_orig, rew_orig, std_orig, best_tree, trees_above_threshold = gather_performance(
        model_path=MODEL_PATH,
        env_name=ENV_NAME,
        transform_func=transform_obs_4meta,
        num_samples=10000,
        n_episodes=20,  # Erhöht von 50 auf 1000
        seeds=list(range(6))  # Erhöht von [0,1,2] auf [0,1,2,3,4,5]
    )

    # Speichere die besten Bäume
    max_num = 0
    pattern = re.compile(r'^run_(\d+)$')
    for entry in os.listdir(OUTPUT_DIR):
        if os.path.isdir(os.path.join(OUTPUT_DIR, entry)):
            match = pattern.match(entry)
            if match:
                folder_num = int(match.group(1))
                max_num = max(max_num, folder_num)
    
    run_folder = f"run_{max_num + 1}"
    TREE_FOLDER = "trees"
    os.makedirs(os.path.join(OUTPUT_DIR, run_folder, TREE_FOLDER), exist_ok=True)
    
    for tree, depth in trees_above_threshold:
        if tree == best_tree:
            filename = os.path.join(OUTPUT_DIR, run_folder, TREE_FOLDER, f"best_tree_depth_{depth}.joblib")
        else:
            filename = os.path.join(OUTPUT_DIR, run_folder, TREE_FOLDER, f"good_tree_depth_{depth}.joblib")
            index += 1
        joblib.dump(tree, filename)

    # Speichere die Performance-Daten
    performance_data = {
        "depths": depths_orig,
        "mean_rewards": rew_orig,
        "std_rewards": std_orig
    }
    performance_filename = os.path.join(OUTPUT_DIR, run_folder, "performance_original_features.joblib")
    joblib.dump(performance_data, performance_filename)

    # Speichere die Performance-Daten als CSV-Datei
    performance_df = pandas.DataFrame(performance_data)
    csv_filename = os.path.join(OUTPUT_DIR, run_folder, "performance_original_features.csv")
    performance_df.to_csv(csv_filename, index=False)

    # Plot der Ergebnisse
    plt.figure(figsize=(8,6))
    plt.errorbar(depths_orig, rew_orig, yerr=std_orig, marker='o', label="Originale Features", capsize=3)
    plt.xlabel("Tree Depth")
    plt.ylabel("Mean Reward (mehrere Seeds x Epis)")
    plt.title("Decision Tree: Mean Reward vs. Max Depth (LunarLander-v2) - Original Features")
    plt.grid(True)
    plt.legend()
    plot_filename = os.path.join(OUTPUT_DIR, run_folder, "mean_reward_original_features.png")
    plt.savefig(plot_filename)
    print(f"Plot gespeichert als '{plot_filename}'.")
    plt.show()

if __name__ == "__main__":
    main()
