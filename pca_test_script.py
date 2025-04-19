import gym
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import random

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from stable_baselines3 import PPO

warnings.filterwarnings("ignore")

# ---------------------------------------
# Konfigurierbare Hyper‑Parameter
# ---------------------------------------
NUM_RANDOM_SEEDS        = 10          # Iterationen in der Suchphase
EPISODES_PER_SEED       = 50          # Datensammlung pro Seed
SEARCH_TREE_DEPTH       = 3           # Baumtiefe während der Seed‑Suche
SEARCH_EVAL_EPISODES    = 30          # Evaluationsepisoden während Suche

# Finalphase
FINAL_TREE_DEPTHS       = range(1, 16)  # Tiefen 1 – 15
FINAL_EVAL_EPISODES     = 100           # Evaluationsepisoden Finalphase
FINAL_RANDOM_SEEDS      = 10            # Wie viele verschiedene Seeds fürs finale Training
TOP_K_FEATURES          = 5             # Anzahl Features nach Selection

# ---------------------------------------
# Dienstprogramme
# ---------------------------------------

def collect_data(env, agent, *, num_episodes, max_steps=1000, seed=None):
    data = []
    for ep in range(num_episodes):
        reset_output = env.reset(seed=None if seed is None else seed + ep)
        state = reset_output[0] if isinstance(reset_output, tuple) else reset_output
        for _ in range(max_steps):
            action, _ = agent.predict(state, deterministic=True)
            step_output = env.step(int(action))
            if len(step_output) == 5:
                next_state, reward, terminated, truncated, _ = step_output
                done = terminated or truncated
            else:
                next_state, reward, done, _ = step_output
            data.append([state, action, reward])
            state = next_state
            if done:
                break
    return pd.DataFrame(data, columns=["state", "action", "reward"])

def preprocess_data(df):
    cols = ["x", "y", "vx", "vy", "theta", "v_theta", "left_leg", "right_leg"]
    state_df = pd.DataFrame(df["state"].tolist(), columns=cols)
    df = pd.concat([state_df, df[["action", "reward"]]], axis=1)
    df["action"].astype(int, copy=False)
    df.dropna(inplace=True)
    return df

def feature_selection(df, *, top_k):
    X = df.drop(["action", "reward"], axis=1)
    y = df["action"].astype(int)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    idx = np.argsort(rf.feature_importances_)[::-1][:top_k]
    return X.iloc[:, idx], X.columns[idx]

def apply_pca(X, *, var_threshold=0.95, seed):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=var_threshold, random_state=seed)
    return pca.fit_transform(X_scaled), scaler, pca

def train_tree(X_tr, y_tr, *, depth, seed):
    tree = DecisionTreeClassifier(max_depth=depth, random_state=seed)
    tree.fit(X_tr, y_tr)
    return tree

def evaluate_tree(tree, scaler, pca, env, sel_cols, *, episodes, max_steps=1000):
    scores = []
    for _ in range(episodes):
        reset_output = env.reset()
        state = reset_output[0] if isinstance(reset_output, tuple) else reset_output
        total = 0
        for _ in range(max_steps):
            df_state = pd.DataFrame([state], columns=["x", "y", "vx", "vy", "theta", "v_theta", "left_leg", "right_leg"])
            state_sel = scaler.transform(df_state[sel_cols])
            state_pca = pca.transform(state_sel)
            action = int(tree.predict(state_pca)[0])
            step_output = env.step(np.clip(action, 0, env.action_space.n - 1))
            if len(step_output) == 5:
                state, reward, terminated, truncated, _ = step_output
                done = terminated or truncated
            else:
                state, reward, done, _ = step_output
            total += reward
            if done:
                break
        scores.append(total)
    return float(np.mean(scores))

# ---------------------------------------
# Hauptablauf
# ---------------------------------------

def main():
    env   = gym.make("LunarLander-v2")
    agent = PPO.load("models/ppo_LunarLander-v2/ppo-LunarLander-v2.zip")

    # ---------------- Suche Phase ----------------
    print("[1/4] Seed‑Suche: pro Seed neue Daten, neue PCA, Baum Tiefe 3 …")
    search_seeds = random.sample(range(1_000_000), NUM_RANDOM_SEEDS)
    best_reward = -np.inf
    best_bundle = None

    for i, seed in enumerate(search_seeds, 1):
        df_raw = collect_data(env, agent, num_episodes=EPISODES_PER_SEED, seed=seed)
        df     = preprocess_data(df_raw)
        X_sel, sel_cols = feature_selection(df, top_k=TOP_K_FEATURES)
        y = df["action"].astype(int)
        X_pca, scaler, pca = apply_pca(X_sel, seed=seed)
        X_tr, _, y_tr, _ = train_test_split(X_pca, y, test_size=0.2, random_state=42)
        tree = train_tree(X_tr, y_tr, depth=SEARCH_TREE_DEPTH, seed=seed)
        mean_r = evaluate_tree(tree, scaler, pca, env, sel_cols, episodes=SEARCH_EVAL_EPISODES)
        print(f" Suche {i:3d}/{NUM_RANDOM_SEEDS}  Seed={seed:6d}  MeanReward={mean_r:6.2f}")
        if mean_r > best_reward:
            best_reward = mean_r
            best_bundle = {
                "seed": seed,
                "X_pca": X_pca,
                "y": y,
                "sel_cols": sel_cols,
                "scaler": scaler,
                "pca": pca
            }

    print(f"\n>>> Bester Seed nach Suche: {best_bundle['seed']}  Reward={best_reward:.2f}")

    # ---------------- Finalphase ----------------
    print("\n[2/4] Finales Training: 10 neue Random‑Seeds, Tiefen 1‑16 …")
    final_seeds = random.sample(range(1_000_000), FINAL_RANDOM_SEEDS)
    sel_cols = best_bundle['sel_cols']
    scaler   = best_bundle['scaler']
    pca      = best_bundle['pca']
    X_tr_all, _, y_tr_all, _ = train_test_split(best_bundle['X_pca'], best_bundle['y'], test_size=0.2, random_state=42)

    results = []
    for fseed in final_seeds:
        for depth in FINAL_TREE_DEPTHS:
            tree   = train_tree(X_tr_all, y_tr_all, depth=depth, seed=fseed)
            mean_r = evaluate_tree(tree, scaler, pca, env, sel_cols, episodes=FINAL_EVAL_EPISODES)
            print(f" Seed {fseed:6d}  Depth {depth:2d}  MeanReward {mean_r:6.2f}")
            results.append([fseed, depth, mean_r])

    env.close()

    # ---------------- Ausgabe ----------------
    print("\n[3/4] Ergebnisse speichern …")
    res_df = pd.DataFrame(results, columns=["Random_Seed", "Tree_Depth", "Mean_Reward"])
    res_df.to_csv("decision_tree_evaluation_final.csv", index=False)

    # Plot: Heatmap‑ähnlich (Lines pro Seed)
    plt.figure(figsize=(10,6))
    for fseed in final_seeds:
        sub = res_df[res_df["Random_Seed"] == fseed]
        plt.plot(sub["Tree_Depth"], sub["Mean_Reward"], marker="o", label=f"Seed {fseed}")
    plt.title("Mean Reward vs. Tree Depth (10 Random Seeds, 100 Eval‑Episoden)")
    plt.xlabel("Tree Depth")
    plt.ylabel("Mean Reward")
    plt.legend()
    plt.grid(True)
    plt.show()

    print("\n[4/4] Fertig – Datei 'decision_tree_evaluation_final.csv' erstellt.")

if __name__ == "__main__":
    main()
