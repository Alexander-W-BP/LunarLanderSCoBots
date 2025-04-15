import argparse
import joblib
import gym
import numpy as np
from enum import Enum
import re
import math

def transform_obs_custom(obs, experiment_name):
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


    if experiment_name == Experiment.ORIGINAL.value:
        return np.array([
            x_space, y_space, vel_x_space, vel_y_space, angle, angular_vel, leg_1, leg_2
        ], dtype=np.float32)
    elif experiment_name == Experiment.PCA_META_FEATURES.value:
        return np.array([
            x_space, y_space, vel_x_space, vel_y_space, angle, angular_vel, leg_1, leg_2, pc2, pc4, pc5
        ], dtype=np.float32)
    elif experiment_name == Experiment.GPT.value:
        return np.array([
            x_space, y_space, vel_x_space, vel_y_space, angle, angular_vel, leg_1, leg_2, speed,
            vel_angle, position_orientation_alignment, position_heading_dot_product, kinetic_energy, rotational_kinetic_energy,
            absolute_angular_vel, angular_acceleration_estimate, horizontal_instability_factor, vertical_landing_readiness,
            distance_to_center
        ], dtype=np.float32)
    elif experiment_name == Experiment.PLOTS_FEATURES_FULL.value:
        return np.array([
            adjusted_angle, adjusted_v_angle, meta_vx_vy, meta_angle_vy, meta_vangle_vy, meta_angle_v_angle,
            x_space, y_space, vel_x_space, vel_y_space, angle, angular_vel, leg_1, leg_2
        ], dtype=np.float32)
    else:
        raise Exception("The features of the following experiment were not defined: " + experiment_name)

def get_experiment_name(model_path):
    # Extract experiment name from path
    match = re.search(r"decision_tree_experiments_([^/\\]+)", model_path)
    if match:
        return match.group(1)
    else:
        raise ValueError("Could not extract experiment name from model_path")

class Experiment(Enum):
    ORIGINAL = "original_features"
    PCA_META_FEATURES = "pca_meta_features"
    GPT = "chat_gpt_features"
    PLOTS_FEATURES_FULL = "plots_features_full"

def main(model_path):

    experiment_name = get_experiment_name(model_path)
    print(f"Running experiment: {experiment_name}")
    print(f"Loading model from: {model_path}")

    # Load the trained DecisionTreeClassifier
    clf = joblib.load(model_path)

    # Create the LunarLander-v2 environment
    env = gym.make('LunarLander-v2', render_mode='human')

    episode = 0
    while True:
        obs, _ = env.reset()
        done = False
        total_reward = 0
        while not done:
            features = transform_obs_custom(obs, experiment_name)
            action = clf.predict([features])[0]
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
        print(f"Episode {episode} reward: {total_reward:.2f}")
        episode += 1

    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a trained DecisionTreeClassifier on LunarLander-v2")
    parser.add_argument("model_path", type=str, help="Path to the saved DecisionTreeClassifier (.joblib)")
    args = parser.parse_args()
    main(args.model_path)
