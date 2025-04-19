from enum import Enum
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_text
import gymnasium as gym
from stable_baselines3 import PPO
import os
import re
import pandas
import joblib
from tqdm import tqdm
import argparse

EXPERIMENT_NAME = None

class Experiment(Enum):
    ORIGINAL = "original_features"
    PCA_META_FEATURES = "pca_meta_features"
    GPT = "chat_gpt_features"
    PLOTS_FEATURES_FULL = "plots_features_full"

def load_model(model_path):
    return PPO.load(model_path)

def transform_obs_custom(obs):
    x_space, y_space, vel_x_space, vel_y_space, angle, angular_vel, leg_1, leg_2 = obs

    # PCA-derived Meta-Features
    pc2 = 0.5 * y_space - 0.5 * vel_y_space
    pc4 = 0.7 * vel_y_space + 0.7 * y_space
    pc5 = 0.7 * vel_x_space - 0.5 * angle - 0.4 * angular_vel

    # Plots-Features
    adjusted_angle = angle - 0.4 * vel_x_space
    adjusted_v_angle = angular_vel - 0.5 * vel_x_space
    meta_vx_vy = vel_y_space - 0.6 * vel_x_space
    meta_angle_vy = angle + 0.8 * vel_y_space
    meta_vangle_vy = angular_vel + 0.8 * vel_y_space
    meta_angle_v_angle = angular_vel + 0.9 * angle

    # ChatGpt recommended features
    epsilon = 1e-8
    speed = math.sqrt(math.pow(vel_x_space, 2) + math.pow(vel_y_space, 2))
    vel_angle = math.atan2(vel_y_space, vel_x_space + epsilon)
    position_orientation_alignment = math.cos(angle)
    position_heading_dot_product = x_space * math.cos(angle) + y_space * math.sin(angle)
    kinetic_energy = 0.5 * (math.pow(vel_x_space, 2) + math.pow(vel_y_space, 2))
    rotational_kinetic_energy = 0.5 * math.pow(angular_vel, 2)
    absolute_angular_vel = abs(angular_vel)
    angular_acceleration_estimate = angular_vel * angle
    horizontal_instability_factor = abs(vel_x_space) + abs(angle)
    vertical_landing_readiness = leg_1 + leg_2
    distance_to_center = math.sqrt(math.pow(x_space, 2) + math.pow(y_space, 2))


    if EXPERIMENT_NAME == Experiment.ORIGINAL.value:
        return np.array([
            x_space, y_space, vel_x_space, vel_y_space, angle, angular_vel, leg_1, leg_2
        ], dtype=np.float32)
    elif EXPERIMENT_NAME == Experiment.PCA_META_FEATURES.value:
        return np.array([
            x_space, y_space, vel_x_space, vel_y_space, angle, angular_vel, leg_1, leg_2, pc2, pc4, pc5
        ], dtype=np.float32)
    elif EXPERIMENT_NAME == Experiment.GPT.value:
        return np.array([
            x_space, y_space, vel_x_space, vel_y_space, angle, angular_vel, leg_1, leg_2, speed,
            vel_angle, position_orientation_alignment, position_heading_dot_product, kinetic_energy, rotational_kinetic_energy,
            absolute_angular_vel, angular_acceleration_estimate, horizontal_instability_factor, vertical_landing_readiness,
            distance_to_center
        ], dtype=np.float32)
    elif EXPERIMENT_NAME == Experiment.PLOTS_FEATURES_FULL.value:
        return np.array([
            adjusted_angle, adjusted_v_angle, meta_vx_vy, meta_angle_vy, meta_vangle_vy, meta_angle_v_angle,
            x_space, y_space, vel_x_space, vel_y_space, angle, angular_vel, leg_1, leg_2
        ], dtype=np.float32)
    else:
        if EXPERIMENT_NAME is None:
             raise Exception("EXPERIMENT_NAME is not set. Ensure --experiment argument is provided.")
        raise Exception("The features of the following experiment were not defined: " + EXPERIMENT_NAME)

def get_tree_text(tree):
    feature_names = []
    if EXPERIMENT_NAME == Experiment.ORIGINAL.value:
        feature_names = [
            "x_space", "y_space", "vel_x_space", "vel_y_space",
            "angle", "angular_vel", "leg_1", "leg_2"
        ]
    elif EXPERIMENT_NAME == Experiment.PCA_META_FEATURES.value:
        feature_names = [
            "x_space", "y_space", "vel_x_space", "vel_y_space",
            "angle", "angular_vel", "leg_1", "leg_2", "pc2", "pc4", "pc5"
        ]
    elif EXPERIMENT_NAME == Experiment.GPT.value:
        feature_names = [
            "x_space", "y_space", "vel_x_space", "vel_y_space",
            "angle", "angular_vel", "leg_1", "leg_2", "speed",
            "vel_angle", "position_orientation_alignment", "position_heading_dot_product",
            "kinetic_energy", "rotational_kinetic_energy", "absolute_angular_vel",
            "angular_acceleration_estimate", "horizontal_instability_factor",
            "vertical_landing_readiness", "distance_to_center"
        ]
    elif EXPERIMENT_NAME == Experiment.PLOTS_FEATURES_FULL.value:
        feature_names = [
            "adjusted_angle", "adjusted_v_angle", "meta_vx_vy", "meta_angle_vy", "meta_vangle_vy", "meta_angle_v_angle",
            "x_space", "y_space", "vel_x_space", "vel_y_space", "angle", "angular_vel", "leg_1", "leg_2"
        ]
    else:
        if EXPERIMENT_NAME is None:
             raise Exception("EXPERIMENT_NAME is not set. Cannot determine feature names.")
        raise Exception("The features of the following experiment were not defined: " + EXPERIMENT_NAME)

    return export_text(tree, feature_names=feature_names)


def evaluate_tree(env, clf, transform_func, n_episodes, max_steps):
    """
    Führt n_episodes lang den Decision Tree in env aus.
    Gibt die Rewards pro Episode zurück. (Liste der totalen Rewards für jede Episode)
    """
    rewards = []
    for ep in tqdm(range(n_episodes), desc="Evaluating Episodes", leave=False):
        obs = env.reset()[0]
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

    return rewards

def gather_performance(model_path, env_name, transform_func,
                       num_samples, n_episodes, seeds):
    """
    Sammelt num_samples Daten mithilfe des PPO-Modells als Orakel
    Trainiert Decision Trees (max_depth=1..15).
    Für jedes Modell und jeden Seed wird evaluate_tree(...) aufgerufen (n_episodes pro Seed).
    Mean Reward wird über alle Episoden berechnet.
    STD wird über die Mean Rewards der einzelnen seeds berechnet.
    Für jede Tiefe wird der entsprechende Baum gespeichert.

    Returns:
        depths (list): List of tree depths tested.
        mean_rewards (list): Mean reward across *all* episodes for each depth.
        seeds_std_rewards (list): Standard deviation of the *per-seed average rewards* for each depth.
        decision_trees_with_depth (list): List of tuples (DecisionTreeClassifier, depth).
    """

    # ---- Daten sammeln mit PPO ----
    env = gym.make(env_name)
    model = load_model(model_path)
    obs_list, act_list = [], []
    obs = env.reset(seed=seeds[0] if seeds else None)[0]
    for _ in tqdm(range(num_samples), desc="Collecting Data", leave=False):
        action, _ = model.predict(obs, deterministic=True)
        obs_list.append(obs)
        act_list.append(action)
        obs, _, done, _, _ = env.step(action)
        if done:
            obs = env.reset(seed=None)[0]
    env.close()

    obs_list = np.array(obs_list)
    act_list = np.array(act_list)

    if transform_func:
        obs_list = np.array([transform_func(o) for o in tqdm(obs_list, desc="Transforming Data", leave=False)])


    depths = range(1, 16)
    mean_rewards = []

    seeds_std_rewards = []
    decision_trees_with_depth = []

    eval_env = gym.make(env_name)

    for depth in tqdm(depths, desc="Training and Evaluating Trees"):
        clf = DecisionTreeClassifier(max_depth=depth, random_state=0)
        clf.fit(obs_list, act_list)

        all_seeds_mean_rewards_list = []
        all_episode_rewards_list = []

        for s in tqdm(seeds, desc=f"Evaluating Depth {depth}", leave=False):
            eval_env.reset(seed=s)
            rewards_for_seed = evaluate_tree(eval_env, clf, transform_func=transform_func,
                                             n_episodes=n_episodes, max_steps=1000)
            all_seeds_mean_rewards_list.append(np.mean(rewards_for_seed))
            all_episode_rewards_list.extend(rewards_for_seed)


        all_episode_rewards_array = np.array(all_episode_rewards_list)
        current_mean_reward = all_episode_rewards_array.mean()

        mean_rewards.append(current_mean_reward)

        all_seeds_mean_rewards_array = np.array(all_seeds_mean_rewards_list)
        seeds_std_rewards.append(all_seeds_mean_rewards_array.std())

        decision_trees_with_depth.append((clf, depth))

    eval_env.close()

    return depths, mean_rewards, seeds_std_rewards, decision_trees_with_depth


def main():
    global EXPERIMENT_NAME

    parser = argparse.ArgumentParser(description="Run an experiment with specified parameters.")

    parser.add_argument(
        "--experiment",
        type=str,
        choices=[e.value for e in Experiment],
        required=True,
        help="Specify the experiment type."
    )
    parser.add_argument(
        "--n_episodes",
        type=int,
        default=100,
        help="Number of evaluation episodes per seed (default: 100)."
    )
    parser.add_argument(
        "--n_seeds",
        type=int,
        default=10,
        help="Number of evaluation seeds (default: 10)."
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10000,
        help="Number of samples to collect using the PPO model (default: 10000)."
    )


    args = parser.parse_args()

    EXPERIMENT_NAME = args.experiment
    N_EPISODES = args.n_episodes
    N_SEEDS = args.n_seeds
    NUM_SAMPLES = args.num_samples

    print(f"Running experiment: {EXPERIMENT_NAME}")
    print(f"Number of evaluation episodes per seed: {N_EPISODES}")
    print(f"Number of evaluation seeds: {N_SEEDS}")
    print(f"Number of samples for data collection: {NUM_SAMPLES}")


    MODEL_PATH = "models/ppo_LunarLander-v2/ppo-LunarLander-v2.zip"
    ENV_NAME = "LunarLander-v2"
    OUTPUT_DIR = f"decision_tree_experiments_{EXPERIMENT_NAME}"

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    depths, rew, seeds_std, decision_trees_with_depth = gather_performance(
        model_path=MODEL_PATH,
        env_name=ENV_NAME,
        transform_func=transform_obs_custom,
        num_samples=NUM_SAMPLES,
        n_episodes=N_EPISODES,
        seeds=list(range(N_SEEDS))
    )


    max_num = 0
    pattern = re.compile(r'^run_(\d+)$')
    if os.path.exists(OUTPUT_DIR):
        for entry in os.listdir(OUTPUT_DIR):
            if os.path.isdir(os.path.join(OUTPUT_DIR, entry)):
                match = pattern.match(entry)
                if match:
                    folder_num = int(match.group(1))
                    max_num = max(max_num, folder_num)

    run_folder_name = f"run_{max_num + 1}"
    run_folder_path = os.path.join(OUTPUT_DIR, run_folder_name)
    TREE_FOLDER_NAME = "trees"
    tree_folder_path = os.path.join(run_folder_path, TREE_FOLDER_NAME)
    os.makedirs(tree_folder_path, exist_ok=True)

    # --- Save Trees ---
    for tree, depth in decision_trees_with_depth:
        tree_text = get_tree_text(tree=tree)
        filename = os.path.join(tree_folder_path, f"tree_depth_{depth}.joblib")
        tree_text_filename = os.path.join(tree_folder_path, f"tree_depth_{depth}.txt")

        joblib.dump(tree, filename)
        with open(tree_text_filename, "w", encoding='utf-8') as f:
            f.write(tree_text)
    print(f"Trees saved in '{tree_folder_path}'.")


    # --- Save Performance Data ---
    performance_data = {
        "depths": depths,
        "mean_rewards_all_episodes": rew,
        "seeds_std_rewards": seeds_std
    }
    performance_filename = os.path.join(run_folder_path, "performance.joblib")
    joblib.dump(performance_data, performance_filename)
    print(f"Performance data saved as '{performance_filename}'.")


    performance_df = pandas.DataFrame(performance_data)
    csv_filename = os.path.join(run_folder_path, "performance.csv")
    performance_df.to_csv(csv_filename, index=False)
    print(f"Performance data saved as '{csv_filename}'.")


    # --- Plotting ---
    plot_filename = os.path.join(run_folder_path, "mean_reward_vs_depth.png")
    plt.figure(figsize=(12, 7))

    x = np.arange(len(depths))
    bar_width = 0.6

    plt.bar(
        x,
        rew, 
        width=bar_width,
        yerr=seeds_std, 
        capsize=4,
        label=f"{EXPERIMENT_NAME} (Error Bars: Std Dev over Seeds)",
        color="#1f77b4"
    )

    plt.xticks(x, depths)
    plt.xlabel("Tree Depth")
    plt.ylabel("Mean Reward (over all episodes)")
    plt.title(f"Decision Tree Performance vs. Max Depth ({ENV_NAME})\nExperiment: {EXPERIMENT_NAME}", fontsize=14)

    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.axhline(y=100, color='grey', linestyle=':', linewidth=1, label='Threshold 100')
    plt.axhline(y=200, color='darkgrey', linestyle=':', linewidth=1, label='Threshold 200')

    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_filename)
    print(f"Plot saved as '{plot_filename}'.")

if __name__ == "__main__":
    main()