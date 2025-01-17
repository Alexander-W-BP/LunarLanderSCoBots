import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
import gymnasium as gym
from stable_baselines3 import PPO

def load_model(model_path):
    return PPO.load(model_path)

def transform_obs_4meta(obs):
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

def evaluate_tree(env, clf, use_meta_features=False, n_episodes=10, max_steps=1000):
    """
    Führt n_episodes lang den Decision Tree in env aus.
    Gibt (mean_reward, std_reward) zurück.
    """
    rewards = []
    for ep in range(n_episodes):
        obs = env.reset()[0]
        total_r = 0.0
        for _ in range(max_steps):
            if use_meta_features:
                obs_ext = transform_obs_4meta(obs)
                action = clf.predict(obs_ext.reshape(1, -1))[0]
            else:
                action = clf.predict(obs.reshape(1, -1))[0]

            obs, reward, done, _, _ = env.step(action)
            total_r += reward
            if done:
                break
        rewards.append(total_r)

    return np.mean(rewards), np.std(rewards)

def gather_performance(model_path, env_name, use_meta_features=False,
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

    if use_meta_features:
        obs_list = np.array([transform_obs_4meta(o) for o in obs_list])

    # -> Hier kein Split, da wir nur Reward messen (oder optional 100% train)
    depths = range(1, 31)
    mean_rewards = []
    std_rewards = []

    eval_env = gym.make(env_name)

    for depth in depths:
        clf = DecisionTreeClassifier(max_depth=depth, random_state=0)
        clf.fit(obs_list, act_list)

        # Mehrere Seeds -> Mittelwert
        all_seeds_rewards = []
        for s in seeds:
            eval_env.reset(seed=s)  # setze seed
            mr, _ = evaluate_tree(eval_env, clf, use_meta_features=use_meta_features,
                                  n_episodes=n_episodes, max_steps=1000)
            all_seeds_rewards.append(mr)

        # Mittelwert und Streuung über seeds
        all_seeds_rewards = np.array(all_seeds_rewards)
        mean_rewards.append(all_seeds_rewards.mean())
        std_rewards.append(all_seeds_rewards.std())

    return depths, mean_rewards, std_rewards

def main():
    MODEL_PATH = "models/ppo-LunarLander-v3/best_model.zip"
    ENV_NAME = "LunarLander-v3"

    # ---- (A) Ohne Meta-Features
    depths_old, rew_old, std_old = gather_performance(
        model_path=MODEL_PATH,
        env_name=ENV_NAME,
        use_meta_features=False,
        num_samples=10000,
        n_episodes=20,   # z.B. 20 Episoden
        seeds=[0, 1, 2]  # und 3 Seeds
    )

    # ---- (B) Mit 4 Meta-Features
    depths_new, rew_new, std_new = gather_performance(
        model_path=MODEL_PATH,
        env_name=ENV_NAME,
        use_meta_features=True,
        num_samples=10000,
        n_episodes=20,
        seeds=[0, 1, 2]
    )

    import matplotlib.pyplot as plt
    plt.figure(figsize=(8,6))
    plt.errorbar(depths_old, rew_old, yerr=std_old, marker='o', label="Ohne Meta-Features", capsize=3)
    plt.errorbar(depths_new, rew_new, yerr=std_new, marker='o', label="Mit Meta-Features", capsize=3)
    plt.xlabel("Tree Depth")
    plt.ylabel("Mean Reward (mehrere Seeds x Epis)")
    plt.title("Decision Tree: Mean Reward vs. Max Depth (LunarLander-v3)")
    plt.grid(True)
    plt.legend()
    plt.savefig("compare_mean_reward_smaller_variance.png")
    plt.show()

if __name__ == "__main__":
    main()
