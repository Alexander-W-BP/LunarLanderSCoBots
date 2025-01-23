import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
import gymnasium as gym
from stable_baselines3 import PPO
import os
import joblib

def load_model(model_path):
    return PPO.load(model_path)

def transform_obs_4meta(obs):
    x_space, y_space, vel_x_space, vel_y_space, angle, angular_vel, leg_1, leg_2 = obs
    # Original features
    return np.array([
        x_space, y_space, vel_x_space, vel_y_space,
        angle, angular_vel, leg_1, leg_2
    ], dtype=np.float32)

def transform_obs_custom(obs):
    x_space, y_space, vel_x_space, vel_y_space, angle, angular_vel, leg_1, leg_2 = obs
    pc2 = 0.5 * y_space - 0.5 * vel_y_space
    pc4 = 0.7 * vel_y_space + 0.7 * y_space
    pc5 = 0.7 * vel_x_space - 0.5 * angle - 0.4 * angular_vel
    return np.array([
        y_space, vel_x_space, vel_y_space, angle, angular_vel, pc2, pc4, pc5
    ], dtype=np.float32)

def evaluate_tree(env, clf, transform_func=None, n_episodes=10, max_steps=1000):
    """
    Führt n_episodes lang den Decision Tree in env aus.
    Gibt (mean_reward, std_reward) zurück.
    """
    rewards = []
    for ep in range(n_episodes):
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

    return np.mean(rewards), np.std(rewards)

def gather_performance(model_path, env_name, transform_func=None,
                       num_samples=10000, n_episodes=10, seeds=[0,1,2,3]):
    """
    - Sammelt num_samples Daten mithilfe des PPO-Modells.
    - Trainiert Decision Trees (max_depth=1..30).
    - Für jedes Modell und jeden Seed wird evaluate_tree(...) aufgerufen (n_episodes pro Seed).
    - Mittelt über alle Seeds => finaler mean Reward + std.
    """
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier

    # ---- Daten sammeln mit PPO ----
    env = gym.make(env_name)
    model = load_model(model_path)
    obs_list, act_list = [], []
    obs = env.reset()[0]
    for _ in range(num_samples):
        action, _ = model.predict(obs, deterministic=True)
        obs_list.append(obs)
        act_list.append(action)
        obs, _, done, _, _ = env.step(action)
        if done:
            obs = env.reset()[0]

    obs_list = np.array(obs_list)
    act_list = np.array(act_list)

    if transform_func:
        obs_list = np.array([transform_func(o) for o in obs_list])

    # -> Hier kein Split, da wir nur Reward messen (oder optional 100% train)
    depths = range(1, 31)
    mean_rewards = []
    std_rewards = []
    best_tree = None
    best_tree_depth = -1

    eval_env = gym.make(env_name)

    for depth in depths:
        clf = DecisionTreeClassifier(max_depth=depth, random_state=0)
        clf.fit(obs_list, act_list)

        # Mehrere Seeds -> Mittelwert
        all_seeds_rewards = []
        for s in seeds:
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
    MODEL_PATH = "models/ppo-LunarLander-v3/best_model.zip" # Passe den Pfad an!
    ENV_NAME = "LunarLander-v3"
    OUTPUT_DIR = "decision_tree_models"

    # Stelle sicher, dass das Ausgabeverzeichnis existiert
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ---- (A) Originale Features
    depths_orig, rew_orig, std_orig, _, _ = gather_performance(
        model_path=MODEL_PATH,
        env_name=ENV_NAME,
        transform_func=transform_obs_4meta,
        num_samples=10000,
        n_episodes=20,
        seeds=[0, 1, 2]
    )

    # ---- (B)  Meta-Features (vel_y_space, y_space, angle, rotational_state, horizontal_state, angular_vel, leg_contact)
    depths_custom, rew_custom, std_custom, best_tree, best_tree_depth = gather_performance(
        model_path=MODEL_PATH,
        env_name=ENV_NAME,
        transform_func=transform_obs_custom,
        num_samples=10000,
        n_episodes=20,
        seeds=[0, 1, 2]
    )

    # Speichere den besten Baum
    if best_tree is not None:
        best_tree_filename = os.path.join(OUTPUT_DIR, f"best_tree_depth_{best_tree_depth}.joblib")
        joblib.dump(best_tree, best_tree_filename)
        print(f"Der beste Baum (Tiefe {best_tree_depth}) wurde in '{best_tree_filename}' gespeichert.")
    else:
        print("Kein Baum mit einem mittleren Reward über 100 gefunden.")

    # Plot der Ergebnisse
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8,6))
    plt.errorbar(depths_orig, rew_orig, yerr=std_orig, marker='o', label="Originale Features", capsize=3)
    plt.errorbar(depths_custom, rew_custom, yerr=std_custom, marker='o', label="Neues Set an Features", capsize=3)
    plt.xlabel("Tree Depth")
    plt.ylabel("Mean Reward (mehrere Seeds x Epis)")
    plt.title("Decision Tree: Mean Reward vs. Max Depth (LunarLander-v3)")
    plt.grid(True)
    plt.legend()
    plt.savefig("compare_mean_reward_original_vs_custom.png")
    plt.show()

if __name__ == "__main__":
    main()