# evaluate_custom_features.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_text
import gymnasium as gym
from stable_baselines3 import PPO
import os
import joblib
import re
import pandas
from tqdm import tqdm  # Importiere tqdm für Fortschrittsbalken

def load_model(model_path):
    return PPO.load(model_path)

def transform_obs_custom(obs):
    x_space, y_space, vel_x_space, vel_y_space, angle, angular_vel, leg_1, leg_2 = obs
    pc2 = 0.5 * y_space - 0.5 * vel_y_space
    pc4 = 0.7 * vel_y_space + 0.7 * y_space
    pc5 = 0.7 * vel_x_space - 0.5 * angle - 0.4 * angular_vel
    return np.array([
        x_space, y_space, vel_x_space, vel_y_space, angle, angular_vel, leg_1, leg_2, pc2, pc4, pc5
    ], dtype=np.float32)

def evaluate_tree(env, clf, transform_func=None, n_episodes=30, max_steps=1000):
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

def gather_performance_custom(model_path, env_name, transform_func=None,
                              num_samples=10000, n_episodes=30, seeds=list(range(6))):
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
    OUTPUT_DIR_ORIG = "decision_tree_models_original"
    OUTPUT_DIR_CUSTOM = "decision_tree_models_custom"

    # Stelle sicher, dass das Ausgabeverzeichnis für custom existiert
    os.makedirs(OUTPUT_DIR_CUSTOM, exist_ok=True)

    # Lade die Performance-Daten der originalen Features
    performance_filename_orig = os.path.join(OUTPUT_DIR_ORIG, "performance_original_features.joblib")
    if not os.path.exists(performance_filename_orig):
        print(f"Die Datei '{performance_filename_orig}' existiert nicht. Bitte führe zuerst das Original-Features-Skript aus.")
        return

    performance_data_orig = joblib.load(performance_filename_orig)
    depths_orig = performance_data_orig["depths"]
    rew_orig = performance_data_orig["mean_rewards"]
    std_orig = performance_data_orig["std_rewards"]

    # ---- Benutzerdefinierte Features
    depths_custom, rew_custom, std_custom, best_tree, trees_above_threshold = gather_performance_custom(
        model_path=MODEL_PATH,
        env_name=ENV_NAME,
        transform_func=transform_obs_custom,
        num_samples=10000,
        n_episodes=500,  # Erhöht von 30 auf 500
        seeds=list(range(6))  # Beibehalten von [0,1,2,3,4,5]
    )

    # Speichere die besten Bäume
    max_num = 0
    pattern = re.compile(r'^run_(\d+)$')
    for entry in os.listdir(OUTPUT_DIR_CUSTOM):
        print(entry)
        if os.path.isdir(os.path.join(OUTPUT_DIR_CUSTOM, entry)):
            match = pattern.match(entry)
            if match:
                folder_num = int(match.group(1))
                max_num = max(max_num, folder_num)
    
    run_folder = f"run_{max_num + 1}"
    TREE_FOLDER = "trees"
    os.makedirs(os.path.join(OUTPUT_DIR_CUSTOM, run_folder, TREE_FOLDER), exist_ok=True)
    
    for tree, depth in trees_above_threshold:
        tree_text = export_text(tree, feature_names=[
            "x_space", "y_space", "vel_x_space", "vel_y_space",
            "angle", "angular_vel", "leg_1", "leg_2", "pc2", "pc4", "pc5"
        ])
    
        if tree == best_tree:
            filename = os.path.join(OUTPUT_DIR_CUSTOM, run_folder, TREE_FOLDER, f"best_tree_depth_{depth}.joblib")
            tree_text_filename = os.path.join(OUTPUT_DIR_CUSTOM, run_folder, TREE_FOLDER, f"best_tree_depth_{depth}.txt")
        else:
            filename = os.path.join(OUTPUT_DIR_CUSTOM, run_folder, TREE_FOLDER, f"good_tree_depth_{depth}.joblib")
            tree_text_filename = os.path.join(OUTPUT_DIR_CUSTOM, run_folder, TREE_FOLDER, f"good_tree_depth_{depth}.txt")
        
        joblib.dump(tree, filename)
        with open(tree_text_filename, "w") as f:
            f.write(tree_text)

    # Speichere die Performance-Daten für custom Features
    performance_data_custom = {
        "depths": depths_custom,
        "mean_rewards": rew_custom,
        "std_rewards": std_custom
    }
    performance_filename_custom = os.path.join(OUTPUT_DIR_CUSTOM, run_folder, "performance_custom_features.joblib")
    joblib.dump(performance_data_custom, performance_filename_custom)

    # Speichere die Performance-Daten als CSV-Datei
    performance_df = pandas.DataFrame(performance_data_custom)
    csv_filename = os.path.join(OUTPUT_DIR_CUSTOM, run_folder, "performance_custom_features.csv")
    performance_df.to_csv(csv_filename, index=False)

    # Plot der Ergebnisse
    plt.figure(figsize=(8,6))
    plt.errorbar(depths_orig, rew_orig, yerr=std_orig, marker='o', label="Originale Features", capsize=3)
    plt.errorbar(depths_custom, rew_custom, yerr=std_custom, marker='s', label="Benutzerdefinierte Features", capsize=3)
    plt.xlabel("Tree Depth")
    plt.ylabel("Mean Reward (mehrere Seeds x Epis)")
    plt.title("Decision Tree: Mean Reward vs. Max Depth (LunarLander-v2)")
    plt.grid(True)
    plt.legend()
    plot_filename = os.path.join(OUTPUT_DIR_CUSTOM, run_folder, "compare_mean_reward_original_vs_custom.png")
    plt.savefig(plot_filename)
    print(f"Vergleichsplot gespeichert als '{plot_filename}'.")
    plt.show()

if __name__ == "__main__":
    main()
