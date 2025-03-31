#!/usr/bin/env python
import os
import numpy as np
import gym
import joblib
from sklearn.tree import export_text
from tqdm import tqdm

def transform_obs_custom(obs):
    """
    Transformiert die Beobachtung in einen Feature-Vektor, der sowohl die Original-Features als auch 
    drei benutzerdefinierte Features (pc2, pc4, pc5) enthält.
    """
    x_space, y_space, vel_x_space, vel_y_space, angle, angular_vel, leg_1, leg_2 = obs
    pc2 = 0.5 * y_space - 0.5 * vel_y_space
    pc4 = 0.7 * vel_y_space + 0.7 * y_space
    pc5 = 0.7 * vel_x_space - 0.5 * angle - 0.4 * angular_vel
    return np.array([
        x_space, y_space, vel_x_space, vel_y_space,
        angle, angular_vel, leg_1, leg_2,
        pc2, pc4, pc5
    ], dtype=np.float32)

def evaluate_tree(env, clf, transform_func, n_episodes, max_steps=1000):
    """
    Führt den Entscheidungsbaum `clf` über n_episodes in der Umgebung aus.
    Falls eine Transformationsfunktion angegeben ist, wird diese auf jede Beobachtung angewendet.
    Gibt den Mittelwert und die Standardabweichung des Rewards zurück.
    """
    rewards = []
    for _ in tqdm(range(n_episodes), desc="Evaluating Episodes"):
        obs, _ = env.reset(seed=None)
        total_reward = 0.0
        for _ in range(max_steps):
            if transform_func:
                obs_trans = transform_func(obs)
                action = clf.predict(obs_trans.reshape(1, -1))[0]
            else:
                action = clf.predict(obs.reshape(1, -1))[0]
            obs, reward, done, truncated, _ = env.step(action)
            total_reward += reward
            if done or truncated:
                break
        rewards.append(total_reward)
    return np.mean(rewards), np.std(rewards)

def main():
    # Spezifischer Pfad zur zu testenden Joblib-Datei
    tree_file_path = os.path.join("decision_tree_models_custom", "run_3", "trees", "best_tree_depth_9.joblib")
    
    if not os.path.exists(tree_file_path):
        print(f"Die Datei '{tree_file_path}' wurde nicht gefunden.")
        return
    
    print(f"Lade Entscheidungsbaum aus: {tree_file_path}")
    best_tree = joblib.load(tree_file_path)
    
    # Monkey-Patch: Falls das Attribut 'monotonic_cst' fehlt, füge es hinzu.
    if not hasattr(best_tree, 'monotonic_cst'):
        best_tree.monotonic_cst = None
    
    # Ausgabe der Baumstruktur (optional)
    tree_text = export_text(
        best_tree, 
        feature_names=[
            "x_space", "y_space", "vel_x_space", "vel_y_space",
            "angle", "angular_vel", "leg_1", "leg_2", "pc2", "pc4", "pc5"
        ]
    )
    print("Baumstruktur des geladenen Entscheidungsbaums:")
    print(tree_text)
    
    # Evaluierung in der Umgebung "LunarLander-v2"
    env = gym.make("LunarLander-v2")
    n_episodes = 500  # Anzahl der zu spielenden Episoden
    mean_reward, std_reward = evaluate_tree(env, best_tree, transform_func=transform_obs_custom, n_episodes=n_episodes)
    
    print("\nEvaluationsergebnisse:")
    print(f"Mean Reward: {mean_reward:.2f}")
    print(f"Standardabweichung: {std_reward:.2f}")

if __name__ == "__main__":
    main()
