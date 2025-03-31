import gym
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
import warnings
import joblib
import os
import re
from tqdm import tqdm
from stable_baselines3 import PPO  # OPTIONAL: Only needed if you're collecting data via a PPO agent

warnings.filterwarnings("ignore")


# =============================================================================
# 1) Datensammlung (OPTIONAL)
# =============================================================================
def collect_data(env, agent, num_episodes=50, max_steps=1000):
    """
    Sammelt Daten von der Interaktion mit einem PPO-Agenten.
    Gibt einen DataFrame mit Spalten ['state', 'action', 'reward'] zurück.
    """
    data = []
    for episode in range(num_episodes):
        reset_output = env.reset()
        if isinstance(reset_output, tuple):
            # Neuere Gym-Versionen
            state, _ = reset_output
        else:
            # Ältere Gym-Versionen
            state = reset_output

        for step in range(max_steps):
            action, _ = agent.predict(state, deterministic=True)
            action = int(action)

            step_output = env.step(action)
            if len(step_output) == 5:
                # Neue Gym API => (next_state, reward, terminated, truncated, info)
                next_state, reward, terminated, truncated, _ = step_output
                done = terminated or truncated
            else:
                # Alte Gym API => (next_state, reward, done, info)
                next_state, reward, done, _ = step_output

            data.append([state, action, reward])
            state = next_state
            if done:
                break

    df = pd.DataFrame(data, columns=["state", "action", "reward"])
    return df


# =============================================================================
# 2) Datenaufbereitung
# =============================================================================
def preprocess_data(df):
    """
    Erwartet einen DataFrame mit Spalte 'state' (Zustandsvektor).
    Erstellt separate Spalten [x, y, vx, vy, theta, v_theta, left_leg, right_leg].
    Konvertiert 'action' in int und entfernt NaNs.
    """
    # Zerlege die State-Vektoren in einzelne Features
    state_features = pd.DataFrame(df["state"].tolist(),
                                  columns=["x", "y", "vx", "vy", "theta",
                                           "v_theta", "left_leg", "right_leg"])
    df = pd.concat([state_features, df[["action", "reward"]]], axis=1)

    # 'action' zu int
    df["action"] = df["action"].astype(int)

    # NaNs entfernen
    init_shape = df.shape
    df.dropna(inplace=True)
    final_shape = df.shape
    print(f"Datenbereinigung: Entfernte {init_shape[0] - final_shape[0]} Zeilen mit fehlenden Werten.")
    return df


# =============================================================================
# 3) Evaluate Decision Tree in Env (Seed & Episode logic)
# =============================================================================
def evaluate_tree_in_env(env, clf, scaler, pca, selected_features,
                         n_episodes=50, max_steps=1000):
    """
    Bewertet einen DecisionTreeClassifier (clf) in der Umgebung (env) 
    für n_episodes Episoden. 
    Gibt eine Liste aller Episoden-Rewards zurück.
    """
    rewards = []
    for _ in range(n_episodes):
        reset_output = env.reset()
        if isinstance(reset_output, tuple):
            state, _ = reset_output
        else:
            state = reset_output

        episode_reward = 0.0
        for _ in range(max_steps):
            # DataFrame für aktuellen Zustand => [x,y,vx,vy,theta,v_theta,left_leg,right_leg]
            state_df = pd.DataFrame([state],
                                    columns=["x", "y", "vx", "vy", "theta",
                                             "v_theta", "left_leg", "right_leg"])
            # Selektiere nur die Spalten, die 'selected_features' vorgibt
            state_sel = state_df[selected_features]
            # Skaliere und wende PCA an
            scaled = scaler.transform(state_sel)
            reduced = pca.transform(scaled)
            # Vorhersage
            action_pred = clf.predict(reduced)[0]
            action = int(np.clip(action_pred, 0, env.action_space.n - 1))

            step_output = env.step(action)
            if len(step_output) == 5:
                next_state, reward, terminated, truncated, _ = step_output
                done = terminated or truncated
            else:
                next_state, reward, done, _ = step_output

            episode_reward += reward
            state = next_state
            if done:
                break

        rewards.append(episode_reward)
    return rewards


# =============================================================================
# 4) Training & Evaluation (Tiefe 1..N, Seeds, etc.)
# =============================================================================
def gather_performance(model_path, env_name, scaler, pca, selected_features,
                       num_samples=10000, n_episodes=100, seeds=list(range(8))):
    """
    Ablauf:
      1) PPO-Daten sammeln (num_samples).
      2) Aufsplitten => X,y (mit 'selected_features'), dann scaler & pca drauf.
      3) Für max_depth in [1..N], DecisionTree trainieren.
      4) Alle Seeds => Evaluate => Mean ± Std
      5) Speichern und Plot

    Parameter:
      - model_path: Pfad zu PPO-Modell
      - env_name: z.B. "LunarLander-v2"
      - scaler, pca: die geladenen Artefakte (joblib)
      - selected_features: Liste der Spaltennamen (aus joblib)
      - num_samples: Wieviele Daten sammeln (Schritte)
      - n_episodes: Wieviele Episoden pro Seed beim Evaluieren
      - seeds: Liste von Seeds, z.B. [0,1,2,3,4]

    Gibt folgende Werte zurück:
      depths, mean_rewards, std_rewards,
      seeds_mean_rewards, seeds_std_rewards,
      best_tree, trees_above_threshold
    """
    from sklearn.tree import DecisionTreeClassifier

    MEAN_REWARD_THRESHOLD = 0

    # 1) Environment & PPO laden
    env = gym.make(env_name)
    model = PPO.load(model_path)

    # 2) PPO-Daten sammeln
    obs_list, act_list = [], []
    obs, _ = env.reset()
    for _ in tqdm(range(num_samples), desc="Sammle Daten"):
        action, _ = model.predict(obs, deterministic=True)
        obs_list.append(obs)
        act_list.append(action)

        step_output = env.step(action)
        if len(step_output) == 5:
            next_state, _, terminated, truncated, _ = step_output
            done = terminated or truncated
        else:
            next_state, _, done, _ = step_output
        obs = next_state
        if done:
            obs, _ = env.reset()

    obs_list = np.array(obs_list)
    act_list = np.array(act_list, dtype=int)

    # 3) DataFrame => Feature engineering
    obs_df = pd.DataFrame(obs_list,
                          columns=["x", "y", "vx", "vy", "theta",
                                   "v_theta", "left_leg", "right_leg"])
    X = obs_df[selected_features].values
    y = act_list

    X_scaled = scaler.transform(X)
    X_pca = pca.transform(X_scaled)

    # 4) Schleife über verschiedene max_depth
    depths = range(1, 8)  # Beispiel: Tiefe 1..7
    mean_rewards = []
    std_rewards = []
    seeds_mean_rewards = []
    seeds_std_rewards = []

    best_tree = None
    best_tree_depth = None
    trees_above_threshold_with_depths = []

    # Fürs Evaluieren brauchen wir ein eval_env:
    eval_env = gym.make(env_name)

    for depth in tqdm(depths, desc="Depths"):
        # DecisionTree trainieren
        dt = DecisionTreeClassifier(max_depth=depth, random_state=0)
        dt.fit(X_pca, y)

        # Evaluate mit seeds
        all_seeds_rewards = []
        all_rewards = []
        for s in tqdm(seeds, desc="Seeds"):
            eval_env.reset(seed=s)
            # Evaluate => liste von Episoden-Rewards
            rewards = []
            for _ in tqdm(range(n_episodes), desc="Episodes"):
                ep_obs, _ = eval_env.reset()
                ep_reward = 0.0
                for _step in range(1000):
                    state_df = pd.DataFrame([ep_obs],
                                            columns=["x","y","vx","vy","theta","v_theta","left_leg","right_leg"])
                    state_sel = state_df[selected_features]
                    scaled = scaler.transform(state_sel)
                    reduced = pca.transform(scaled)
                    action_pred = dt.predict(reduced)[0]
                    action = int(np.clip(action_pred, 0, eval_env.action_space.n - 1))

                    step_out = eval_env.step(action)
                    if len(step_out) == 5:
                        next_obs, r, terminated, truncated, _ = step_out
                        done_ = terminated or truncated
                    else:
                        next_obs, r, done_, _ = step_out

                    ep_reward += r
                    ep_obs = next_obs
                    if done_:
                        break
                rewards.append(ep_reward)

            # Mittelwert pro Seed
            seed_mean = np.mean(rewards)
            all_seeds_rewards.append(seed_mean)
            # Alle Episoden in "all_rewards"
            all_rewards.extend(rewards)

        # Gesamt:
        all_rewards = np.array(all_rewards)
        cur_mean = all_rewards.mean()
        cur_std = all_rewards.std()
        mean_rewards.append(cur_mean)
        std_rewards.append(cur_std)

        # Seeds => Mittelwert & Std
        all_seeds_rewards = np.array(all_seeds_rewards)
        seeds_mean_rewards.append(all_seeds_rewards.mean())
        seeds_std_rewards.append(all_seeds_rewards.std())

        # Threshold / Bester Baum
        if cur_mean > MEAN_REWARD_THRESHOLD and (best_tree is None or depth < best_tree_depth):
            best_tree = dt
            best_tree_depth = depth

        if cur_mean > MEAN_REWARD_THRESHOLD:
            trees_above_threshold_with_depths.append((dt, depth))

        print(f"Depth={depth}: Mean Reward={cur_mean:.2f} ± {cur_std:.2f}")

    return (depths,
            mean_rewards,
            std_rewards,
            seeds_mean_rewards,
            seeds_std_rewards,
            best_tree,
            trees_above_threshold_with_depths)


# =============================================================================
# MAIN
# =============================================================================
def main():
    # 1) Parameter
    MODEL_PATH = "models/ppo_LunarLander-v2/ppo-LunarLander-v2.zip"
    ENV_NAME = "LunarLander-v2"
    OUTPUT_DIR = "decision_tree_experiment"

    # 2) Lade PCA/Scaler/Features
    folder_artifacts = "decision_trees_best"  # Pfad zu den Joblib-Artefakten
    scaler_path = os.path.join(folder_artifacts, "scaler.joblib")
    pca_path = os.path.join(folder_artifacts, "pca.joblib")
    feats_path = os.path.join(folder_artifacts, "selected_features.joblib")

    scaler = joblib.load(scaler_path)
    pca = joblib.load(pca_path)
    selected_features = joblib.load(feats_path)
    print("Geladene Artefakte:", scaler, pca, selected_features, sep="\n")

    # 3) Evtl. Ordner anlegen
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 4) Sammle Data, trainiere Bäume & evaluiere
    depths, rew, std, seeds_mean, seeds_std, best_tree, good_trees = gather_performance(
        model_path=MODEL_PATH,
        env_name=ENV_NAME,
        scaler=scaler,
        pca=pca,
        selected_features=selected_features,
        num_samples=10000,
        n_episodes=100,      # z.B. 10 Episoden pro Seed
        seeds=list(range(100))  # z.B. Seeds 0..4
    )

    # 5) Ergebnisse speichern
    max_run = 0
    pattern = re.compile(r"^run_(\d+)$")
    for entry in os.listdir(OUTPUT_DIR):
        if os.path.isdir(os.path.join(OUTPUT_DIR, entry)):
            match = pattern.match(entry)
            if match:
                folder_num = int(match.group(1))
                if folder_num > max_run:
                    max_run = folder_num
    run_folder = f"run_{max_run + 1}"
    run_path = os.path.join(OUTPUT_DIR, run_folder)
    os.makedirs(run_path, exist_ok=True)

    # 6) Performance-Daten
    perf_data = {
        "depths": depths,
        "mean_rewards": rew,
        "std_rewards": std,
        "seeds_mean_rewards": seeds_mean,
        "seeds_std_rewards": seeds_std
    }
    joblib.dump(perf_data, os.path.join(run_path, "performance.joblib"))

    df_perf = pd.DataFrame(perf_data)
    df_perf.to_csv(os.path.join(run_path, "performance.csv"), index=False)

    # 7) Plot
    plt.figure(figsize=(8, 6))
    plt.errorbar(depths, rew, yerr=std, marker='o', capsize=3,
                 label="Mean Reward ± Std (All seeds & episodes)")
    plt.xlabel("Tree Depth")
    plt.ylabel("Mean Reward")
    plt.title("Decision Tree: Mean Reward vs. Depth (mit Seeds/Episoden)")
    plt.grid(True)
    plt.legend()
    plot_path = os.path.join(run_path, "mean_reward_vs_depth.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Plot gespeichert in '{plot_path}'.")


if __name__ == "__main__":
    main()
