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
    speed = math.sqrt(math.pow(vel_x_space, 2) + math.pow(vel_y_space, 2))
    vel_angle = math.atan(vel_y_space/vel_x_space)
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
        raise Exception("The features of the following experiment were not defined: " + EXPERIMENT_NAME)

def get_tree_text(tree):
    if EXPERIMENT_NAME == Experiment.ORIGINAL.value:
        return export_text(tree, feature_names=[
            "x_space", "y_space", "vel_x_space", "vel_y_space",
            "angle", "angular_vel", "leg_1", "leg_2"
        ])
    elif EXPERIMENT_NAME == Experiment.PCA_META_FEATURES.value:
        return export_text(tree, feature_names=[
            "x_space", "y_space", "vel_x_space", "vel_y_space",
            "angle", "angular_vel", "leg_1", "leg_2", "pc2", "pc4", "pc5"
        ])
    elif EXPERIMENT_NAME == Experiment.GPT.value:
        return export_text(tree, feature_names=[
            "x_space", "y_space", "vel_x_space", "vel_y_space",
            "angle", "angular_vel", "leg_1", "leg_2", "speed",
            "vel_angle", "position_orientation_alignment", "position_heading_dot_product",
            "kinetic_energy", "rotational_kinetic_energy", "absolute_angular_vel",
            "angular_acceleration_estimate", "horizontal_instability_factor",
            "vertical_landing_readiness", "distance_to_center"
        ])
    elif EXPERIMENT_NAME == Experiment.PLOTS_FEATURES_FULL.value:
        return export_text(tree, feature_names=[
            "adjusted_angle", "adjusted_v_angle", "meta_vx_vy", "meta_angle_vy", "meta_vangle_vy", "meta_angle_v_angle",
            "x_space", "y_space", "vel_x_space", "vel_y_space", "angle", "angular_vel", "leg_1", "leg_2"
        ])
    else:
        raise Exception("The features of the following experiment were not defined: " + EXPERIMENT_NAME)

def evaluate_tree(env, clf, transform_func, n_episodes, max_steps):
    """
    Führt n_episodes lang den Decision Tree in env aus.
    Gibt die Mean Rewards pro episode zurück.
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

    return rewards

def gather_performance(model_path, env_name, transform_func,
                       num_samples, n_episodes, seeds):
    """
    Sammelt num_samples Daten mithilfe des PPO-Modells als Orakel
    Trainiert Decision Trees (max_depth=1..15).
    Für jedes Modell und jeden Seed wird evaluate_tree(...) aufgerufen (n_episodes pro Seed).
    Mean Reward und STD wird sowohl über alle Episoden berechnet als auch über die Mean Rewards der einzelnen seeds.
    Für jede Tiefe wird der beste Baum mit einem Reward über einem bestimmten Threshold gespeichert

    """

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


    depths = range(1, 16)
    mean_rewards = []
    std_rewards = []
    seeds_mean_rewards = []
    seeds_std_rewards = []
    decision_trees_with_depth = []

    eval_env = gym.make(env_name)

    for depth in tqdm(depths, desc="Training Trees"):
        clf = DecisionTreeClassifier(max_depth=depth, random_state=0)
        clf.fit(obs_list, act_list)

        all_seeds_rewards = []
        all_rewards = []
        for s in tqdm(seeds, desc=f"Evaluating Depth {depth}", leave=False):
            eval_env.reset(seed=s)  
            rewards = evaluate_tree(eval_env, clf, transform_func=transform_func,
                                 n_episodes=n_episodes, max_steps=1000)
            all_seeds_rewards.append(np.mean(rewards))
            all_rewards.extend(rewards)


        all_rewards = np.array(all_rewards)
        current_mean_reward = all_rewards.mean()
        current_std_reward = all_rewards.std()

        mean_rewards.append(current_mean_reward)
        std_rewards.append(current_std_reward)

        all_seeds_rewards = np.array(all_seeds_rewards)
        seeds_mean_rewards.append(all_seeds_rewards.mean())
        seeds_std_rewards.append(all_seeds_rewards.std())

        decision_trees_with_depth.append((clf, depth))

    return depths, mean_rewards, std_rewards, seeds_mean_rewards, seeds_std_rewards, decision_trees_with_depth

class Experiment(Enum):
    ORIGINAL = "original_features"
    PCA_META_FEATURES = "pca_meta_features"
    GPT = "chat_gpt_features"
    PLOTS_FEATURES_FULL = "plots_features_full"


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
        help="Number of episodes (default: 100)."
)
parser.add_argument(
    "--n_seeds",
    type=int,
    default=100,  
    help="Number of seeds (default: 10)."
)

args = parser.parse_args()

# Set variables
EXPERIMENT_NAME = args.experiment
N_EPISODES = args.n_episodes
N_SEEDS = args.n_seeds

print(f"Running experiment: {EXPERIMENT_NAME}")
print(f"Number of episodes: {N_EPISODES}")
print(f"Number of seeds: {N_SEEDS}")


def main():
    MODEL_PATH = "models/ppo_LunarLander-v2/ppo-LunarLander-v2.zip" 
    ENV_NAME = "LunarLander-v2"
    OUTPUT_DIR = "decision_tree_experiments_" + EXPERIMENT_NAME

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    depths, rew, std, seeds_rew, seeds_std, decision_trees_with_depth = gather_performance(
        model_path=MODEL_PATH,
        env_name=ENV_NAME,
        transform_func=transform_obs_custom,
        num_samples=10000,
        n_episodes=N_EPISODES, 
        seeds=list(range(N_SEEDS))
    )


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

    for tree, depth in decision_trees_with_depth:
        tree_text = get_tree_text(tree=tree)
        filename = os.path.join(OUTPUT_DIR, run_folder, TREE_FOLDER, f"tree_depth_{depth}.joblib")
        tree_text_filename = os.path.join(OUTPUT_DIR, run_folder, TREE_FOLDER, f"tree_depth_{depth}.txt")

        joblib.dump(tree, filename)
        with open(tree_text_filename, "w") as f:
            f.write(tree_text)


    performance_data = {
        "depths": depths,
        "mean_rewards": rew,
        "std_rewards": std,
        "seeds_mean_rewards": seeds_rew,
        "seeds_std_rewards": seeds_std
    }
    performance_filename = os.path.join(OUTPUT_DIR, run_folder, "performance.joblib")
    joblib.dump(performance_data, performance_filename)


    performance_df = pandas.DataFrame(performance_data)
    csv_filename = os.path.join(OUTPUT_DIR, run_folder, "performance.csv")
    performance_df.to_csv(csv_filename, index=False)


    plt.figure(figsize=(8,6))
    plt.errorbar(depths, rew, yerr=std, marker='o', label=EXPERIMENT_NAME, capsize=3)
    plt.xlabel("Tree Depth")
    plt.ylabel("Mean Reward (over all seeds and episodes)")
    plt.title("Decision Tree: Mean Reward vs. Max Depth (LunarLander-v2) -" + EXPERIMENT_NAME)
    plt.grid(True)
    plt.legend()
    plot_filename = os.path.join(OUTPUT_DIR, run_folder, "mean_reward_vs_depth.png")
    plt.savefig(plot_filename)
    print(f"Plot gespeichert als '{plot_filename}'.")

if __name__ == "__main__":
    main()
