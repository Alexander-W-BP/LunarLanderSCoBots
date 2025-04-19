"""lunar_lander_pca_decision_tree_cli.py  (updated)

Command‑line friendly version of the PCA + Decision‑Tree pipeline.
Creates a fresh results folder for every run (run_1, run_2, …) and stores:
  • configuration.json (all CLI parameters)
  • best_seed.txt (seed + search reward)
  • decision_tree_evaluation_final.csv (results table)
  • mean_reward_vs_depth.png (plot)
  • trees/ → text dumps of every Decision‑Tree from the final phase
"""

import argparse
import json
import os
import random
import re
import warnings

import gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text
from stable_baselines3 import PPO

warnings.filterwarnings("ignore")

# ───────────────────────────────────────────────────────── helper ──────────


def collect_data(env, agent, *, episodes: int, max_steps: int = 1000, seed: int | None = None):
    data = []
    for ep in range(episodes):
        s_out = env.reset(seed=None if seed is None else seed + ep)
        state = s_out[0] if isinstance(s_out, tuple) else s_out
        for _ in range(max_steps):
            action, _ = agent.predict(state, deterministic=True)
            step = env.step(int(action))
            next_state = step[0] if len(step) == 5 else step[0]
            done = step[2] if len(step) == 5 else step[2]
            data.append((state, action))
            state = next_state
            if done:
                break
    return data


def preprocess_data(records):
    cols = ["x", "y", "vx", "vy", "theta", "v_theta", "left_leg", "right_leg"]
    states, actions = zip(*records)
    return pd.DataFrame(list(states), columns=cols), pd.Series(actions, name="action", dtype=int)


def feature_selection(X, y, *, k: int):
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    idx = np.argsort(rf.feature_importances_)[::-1][:k]
    return X.iloc[:, idx], X.columns[idx]


def apply_pca(X, *, var=0.95, seed: int):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=var, random_state=seed)
    return pca.fit_transform(X_scaled), scaler, pca


def train_tree(X_tr, y_tr, *, depth: int, seed: int):
    clf = DecisionTreeClassifier(max_depth=depth, random_state=seed)
    clf.fit(X_tr, y_tr)
    return clf


def evaluate_tree(tree, scaler, pca, env, feature_cols, *, episodes: int, max_steps: int = 1000):
    rewards = []
    for _ in range(episodes):
        st = env.reset()[0]
        total = 0
        for _ in range(max_steps):
            df = pd.DataFrame([st], columns=["x", "y", "vx", "vy", "theta", "v_theta", "left_leg", "right_leg"])
            st_scaled = scaler.transform(df[feature_cols])
            st_pca = pca.transform(st_scaled)
            act = int(tree.predict(st_pca)[0])
            out = env.step(np.clip(act, 0, env.action_space.n - 1))
            st = out[0]
            total += out[1]
            if out[2]:  # done flag
                break
        rewards.append(total)
    return float(np.mean(rewards))

# ───────────────────────────────────────────────────────── CLI ─────────────

parser = argparse.ArgumentParser(description="PCA + Decision‑Tree benchmark for LunarLander‑v2")
parser.add_argument("--search_seeds", type=int, default=10)
parser.add_argument("--episodes_per_seed", type=int, default=50)
parser.add_argument("--search_eval_episodes", type=int, default=30)
parser.add_argument("--final_seeds", type=int, default=10)
parser.add_argument("--final_eval_episodes", type=int, default=100)
parser.add_argument("--max_depth", type=int, default=15)
parser.add_argument("--model_path", type=str, default="models/ppo_LunarLander-v2/ppo-LunarLander-v2.zip")
parser.add_argument("--output_dir", type=str, default="pca_dt_runs")
parser.add_argument("--top_k", type=int, default=5)
ARGS = parser.parse_args()

# ───────────────────────────────────────────────────────── main ────────────

def main():
    # create new run folder
    os.makedirs(ARGS.output_dir, exist_ok=True)
    idx = 1
    for f in os.listdir(ARGS.output_dir):
        m = re.fullmatch(r"run_(\d+)", f)
        if m:
            idx = max(idx, int(m.group(1)) + 1)
    RUN = os.path.join(ARGS.output_dir, f"run_{idx}")
    TREE_DIR = os.path.join(RUN, "trees")
    os.makedirs(TREE_DIR, exist_ok=True)

    # save config
    with open(os.path.join(RUN, "configuration.json"), "w") as fp:
        json.dump(vars(ARGS), fp, indent=2)

    env = gym.make("LunarLander-v2")
    agent = PPO.load(ARGS.model_path)

    # ───── search phase
    best_seed, best_reward, best_bundle = None, -np.inf, None
    for s in random.sample(range(1_000_000), ARGS.search_seeds):
        rec = collect_data(env, agent, episodes=ARGS.episodes_per_seed, seed=s)
        X_raw, y = preprocess_data(rec)
        X_sel, sel_cols = feature_selection(X_raw, y, k=ARGS.top_k)
        X_pca, scaler, pca = apply_pca(X_sel, seed=s)
        X_tr, _, y_tr, _ = train_test_split(X_pca, y, test_size=0.2, random_state=42)
        tree = train_tree(X_tr, y_tr, depth=3, seed=s)
        r = evaluate_tree(tree, scaler, pca, env, sel_cols, episodes=ARGS.search_eval_episodes)
        if r > best_reward:
            best_seed, best_reward = s, r
            best_bundle = dict(sel_cols=sel_cols, scaler=scaler, pca=pca, X_pca=X_pca, y=y)
    with open(os.path.join(RUN, "best_seed.txt"), "w") as f:
        f.write(f"seed: {best_seed}\nmean_reward: {best_reward:.2f}\n")

    # ───── final phase
    depths = range(1, ARGS.max_depth + 1)
    final_seeds = random.sample(range(1_000_000), ARGS.final_seeds)
    results = []

    X_pca_best, y_best = best_bundle["X_pca"], best_bundle["y"]
    sel_cols, scaler, pca = best_bundle["sel_cols"], best_bundle["scaler"], best_bundle["pca"]
    X_tr, _, y_tr, _ = train_test_split(X_pca_best, y_best, test_size=0.2, random_state=42)

    feat_names = [f"PC{i+1}" for i in range(X_tr.shape[1])]

    for fseed in final_seeds:
        for d in depths:
            tree = train_tree(X_tr, y_tr, depth=d, seed=fseed)
            m_reward = evaluate_tree(tree, scaler, pca, env, sel_cols, episodes=ARGS.final_eval_episodes)
            results.append((fseed, d, m_reward))

            # save tree text
            text_path = os.path.join(TREE_DIR, f"tree_seed{fseed}_depth{d}.txt")
            with open(text_path, "w") as fp:
                fp.write(export_text(tree, feature_names=feat_names))

    env.close()

    # results csv / plot
    df = pd.DataFrame(results, columns=["random_seed", "tree_depth", "mean_reward"])
    df.to_csv(os.path.join(RUN, "decision_tree_evaluation_final.csv"), index=False)

    plt.figure(figsize=(10, 6))
    for fseed in final_seeds:
        sub = df[df.random_seed == fseed]
        plt.plot(sub.tree_depth, sub.mean_reward, marker="o", label=f"seed {fseed}")
    plt.title("Mean Reward vs Tree Depth – final phase")
    plt.xlabel("Tree depth")
    plt.ylabel("Mean reward")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(RUN, "mean_reward_vs_depth.png"))

    print("✓ Saved all results in", RUN)


if __name__ == "__main__":
    main()
