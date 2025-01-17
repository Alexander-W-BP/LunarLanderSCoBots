"""
Vergleicht zwei Varianten von Decision Trees (ohne vs. mit 4 Meta-Features)
nur anhand des Mean Rewards auf LunarLander-v3.
Erzeugt eine Grafik mit zwei Linien (Fehlerbalken):
 - Blau: Original-Features (8D)
 - Rot:  Mit 4 Meta-Features (12D)
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Falls du kein GUI-Fenster öffnen willst
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from stable_baselines3 import PPO
import gymnasium as gym

# -----------------------------------------------------
# 1) PPO-Modell laden
# -----------------------------------------------------
def load_model(model_path):
    """
    Lädt ein trainiertes PPO-Modell.
    """
    model = PPO.load(model_path)
    return model

# -----------------------------------------------------
# 2) Meta-Features erstellen
# -----------------------------------------------------
def transform_obs_4meta(obs):
    """
    Erzeugt für eine 8D-Beobachtung 4 Meta-Features
    (rotational_state, horizontal_state, vertical_state, leg_contact).
    Ergebnis: 12D.
    """
    x_space, y_space, vel_x_space, vel_y_space, angle, angular_vel, leg_1, leg_2 = obs

    rotational_state = angle + 0.5 * angular_vel
    horizontal_state = x_space + 0.5 * vel_x_space
    vertical_state   = y_space + 0.5 * vel_y_space
    leg_contact      = leg_1 + leg_2

    return np.array([
        x_space, y_space, vel_x_space, vel_y_space,
        angle, angular_vel, leg_1, leg_2,
        rotational_state, horizontal_state, vertical_state, leg_contact
    ], dtype=np.float32)

# -----------------------------------------------------
# 3) Evaluate-Funktion (Mean Reward)
# -----------------------------------------------------
def evaluate_tree(env, clf, use_meta_features=False, n_episodes=10, max_steps=1000):
    """
    Lässt den Decision Tree clf n_episodes lang in der Umgebung env agieren
    und sammelt dabei den Reward jeder Episode.
    use_meta_features=False -> wir nutzen die 8D-Obs direkt
    use_meta_features=True  -> wir hängen 4 Meta-Features an (->12D)
    Gibt (mean_reward, std_reward) zurück.
    """
    rewards = []
    for ep in range(n_episodes):
        obs = env.reset()[0]  # 8D-Observation
        ep_reward = 0.0
        for _ in range(max_steps):
            if use_meta_features:
                obs_extended = transform_obs_4meta(obs)
                action = clf.predict(obs_extended.reshape(1, -1))[0]
            else:
                # Nur die 8D-Obs
                action = clf.predict(obs.reshape(1, -1))[0]

            obs, reward, done, _, _ = env.step(action)
            ep_reward += reward
            if done:
                break
        rewards.append(ep_reward)

    return np.mean(rewards), np.std(rewards)

# -----------------------------------------------------
# 4) gather_performance (ohne Accuracy)
# -----------------------------------------------------
def gather_performance(model_path, env_name, num_samples=10000, use_meta_features=False):
    """
    Sammelt mit PPO deterministisch Daten (8D-Observations).
    Wenn use_meta_features=True, wandeln wir jeden obs in 12D um.
    Für max_depth=1..30 trainieren wir je einen Tree (auf ALLEN Daten, kein Split)
    und messen den Mean Reward in der Env.
    Gibt depths, mean_rewards, std_rewards zurück.
    """

    # a) Environment & PPO laden
    env = gym.make(env_name)
    model = load_model(model_path)

    # b) Daten sammeln
    observations, actions = [], []
    obs = env.reset()[0]
    for _ in range(num_samples):
        action, _ = model.predict(obs, deterministic=True)
        observations.append(obs)
        actions.append(action)
        obs, _, done, _, _ = env.step(action)
        if done:
            obs = env.reset()[0]

    observations = np.array(observations)  # shape (num_samples, 8)
    actions = np.array(actions)

    # c) Optional: in 12D umwandeln
    if use_meta_features:
        obs_extended = np.array([transform_obs_4meta(o) for o in observations])
    else:
        obs_extended = observations

    # d) Eval-Env erstellen
    eval_env = gym.make(env_name)

    depths = range(1, 31)
    reward_list = []
    reward_std_list = []

    # e) Für jede Tiefe trainieren & Reward messen
    for depth in depths:
        clf = DecisionTreeClassifier(max_depth=depth, random_state=0)
        # Trainiere auf GESAMTEN Daten (kein Split, da wir keine Accuracy wollen)
        clf.fit(obs_extended, actions)

        # Evaluate Mean Reward
        mean_rew, std_rew = evaluate_tree(eval_env, clf, use_meta_features=use_meta_features,
                                          n_episodes=10, max_steps=1000)
        reward_list.append(mean_rew)
        reward_std_list.append(std_rew)

    return depths, reward_list, reward_std_list

# -----------------------------------------------------
# 5) main: Vergleicht Original (8D) vs. 4 Meta-Features (12D)
# -----------------------------------------------------
def main():
    MODEL_PATH = "models/ppo-LunarLander-v3/best_model.zip"
    ENV_NAME = "LunarLander-v3"
    NUM_SAMPLES = 10000

    print("\n=== Vergleich OHNE vs. MIT 4 Meta-Features (nur Mean Reward) ===")

    print("\n--- (A) Ohne Meta-Features ---")
    depths_old, rew_old, std_old = gather_performance(
        model_path=MODEL_PATH,
        env_name=ENV_NAME,
        num_samples=NUM_SAMPLES,
        use_meta_features=False
    )

    print("\n--- (B) Mit 4 Meta-Features ---")
    depths_new, rew_new, std_new = gather_performance(
        model_path=MODEL_PATH,
        env_name=ENV_NAME,
        num_samples=NUM_SAMPLES,
        use_meta_features=True
    )

    # 6) Plot: In einer Grafik nur den Mean Reward (mit Fehlerbalken)
    plt.figure(figsize=(8, 6))
    plt.errorbar(depths_old, rew_old, yerr=std_old, marker='o', color='blue', capsize=3, label="Ohne Meta-Features")
    plt.errorbar(depths_new, rew_new, yerr=std_new, marker='o', color='red',  capsize=3, label="Mit Meta-Features")
    plt.xlabel("Tree Depth")
    plt.ylabel("Mean Reward (10 Epis)")
    plt.title("Decision Tree: Mean Reward vs. Max Depth (LunarLander-v3)")
    plt.grid(True)
    plt.legend(loc="best")
    plt.savefig("compare_old_vs_new_features_mean_reward.png")
    plt.show()

    print("Fertig! Die Grafik 'compare_old_vs_new_features_mean_reward.png' wurde erstellt.")

if __name__ == "__main__":
    main()
