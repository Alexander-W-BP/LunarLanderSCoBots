# train_original_features.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
import gymnasium as gym
from stable_baselines3 import PPO
import os
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
        if current_mean_reward > 100 and (best_tree is None or depth < best_tree_depth):
            best_tree = clf
            best_tree_depth = depth

    return depths, mean_rewards, std_rewards, best_tree, best_tree_depth

def main():
    MODEL_PATH = "models/ppo-LunarLander-v3/best_model.zip"  # Passe den Pfad an!
    ENV_NAME = "LunarLander-v3"
    OUTPUT_DIR = "decision_tree_models_original"

    # Stelle sicher, dass das Ausgabeverzeichnis existiert
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ---- Originale Features
    depths_orig, rew_orig, std_orig, best_tree, best_tree_depth = gather_performance(
        model_path=MODEL_PATH,
        env_name=ENV_NAME,
        transform_func=transform_obs_4meta,
        num_samples=10000,
        n_episodes=50,  # Erhöht von 20 auf 50
        seeds=list(range(6))  # Erhöht von [0,1,2] auf [0,1,2,3,4,5]
    )

    # Speichere den besten Baum
    if best_tree is not None:
        best_tree_filename = os.path.join(OUTPUT_DIR, f"best_tree_depth_{best_tree_depth}.joblib")
        joblib.dump(best_tree, best_tree_filename)
        print(f"Der beste Baum (Tiefe {best_tree_depth}) wurde in '{best_tree_filename}' gespeichert.")
    else:
        print("Kein Baum mit einem mittleren Reward über 100 gefunden.")

    # Speichere die Performance-Daten
    performance_data = {
        "depths": depths_orig,
        "mean_rewards": rew_orig,
        "std_rewards": std_orig
    }
    performance_filename = os.path.join(OUTPUT_DIR, "performance_original_features.joblib")
    joblib.dump(performance_data, performance_filename)
    print(f"Performance-Daten gespeichert in '{performance_filename}'.")

    # Plot der Ergebnisse
    plt.figure(figsize=(8,6))
    plt.errorbar(depths_orig, rew_orig, yerr=std_orig, marker='o', label="Originale Features", capsize=3)
    plt.xlabel("Tree Depth")
    plt.ylabel("Mean Reward (mehrere Seeds x Epis)")
    plt.title("Decision Tree: Mean Reward vs. Max Depth (LunarLander-v3) - Original Features")
    plt.grid(True)
    plt.legend()
    plot_filename = os.path.join(OUTPUT_DIR, "mean_reward_original_features.png")
    plt.savefig(plot_filename)
    print(f"Plot gespeichert als '{plot_filename}'.")
    plt.show()

if __name__ == "__main__":
    main()
