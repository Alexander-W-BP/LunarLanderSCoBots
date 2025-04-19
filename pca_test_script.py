"""lunar_lander_pca_decision_tree_cli.py

Command‑line friendly version of the PCA + Decision‑Tree pipeline.
Creates a fresh results folder for every run (run_1, run_2, …) and
stores:
  • configuration.json (all CLI parameters)
  • best_seed.txt (seed + search reward)
  • decision_tree_evaluation_final.csv (depth → mean reward, aggregated across evaluation seeds)
  • mean_reward_vs_depth.png (plot)
  • decision_trees/            (one *.joblib ***und*** eine *.txt‑Darstellung pro Baumtiefe)
  • search_phase_trees/        (je Seed aus der Suchphase eine *.txt‑Darstellung)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import random
import warnings

import gym
import joblib  # zum Persistieren der finalen Modelle
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

# ---------------------------------------------------------------------------
# ---------------------------- Helper functions -----------------------------
# ---------------------------------------------------------------------------

def collect_data(env, agent, *, episodes: int, max_steps: int = 1000, seed: int | None = None):
    """Collects (state, action) pairs from a PPO agent acting greedily."""
    data = []
    for ep in range(episodes):
        reset_out = env.reset(seed=None if seed is None else seed + ep)
        state = reset_out[0] if isinstance(reset_out, tuple) else reset_out
        for _ in range(max_steps):
            action, _ = agent.predict(state, deterministic=True)
            step_out = env.step(int(action))
            if len(step_out) == 5:
                next_state, reward, terminated, truncated, _ = step_out
                done = terminated or truncated
            else:  # gym (not gymnasium) fallback
                next_state, reward, done, _ = step_out
            data.append((state, action))
            state = next_state
            if done:
                break
    return data

def preprocess_data(records):
    """Turns list of (state, action) into X, y DataFrames."""
    cols = ["x", "y", "vx", "vy", "theta", "v_theta", "left_leg", "right_leg"]
    states, actions = zip(*records)
    X = pd.DataFrame(list(states), columns=cols)
    y = pd.Series(actions, name="action").astype(int)
    return X, y

def feature_selection(X: pd.DataFrame, y: pd.Series, *, k: int, rf_seed: int = 42):
    rf = RandomForestClassifier(n_estimators=100, random_state=rf_seed)
    rf.fit(X, y)
    best_idx = np.argsort(rf.feature_importances_)[::-1][:k]
    return X.iloc[:, best_idx], X.columns[best_idx]

def apply_pca(X: pd.DataFrame, *, variance: float, seed: int):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=variance, random_state=seed)
    X_pca = pca.fit_transform(X_scaled)
    return X_pca, scaler, pca

def train_tree(X_tr, y_tr, *, depth: int, seed: int):
    tree = DecisionTreeClassifier(max_depth=depth, random_state=seed)
    tree.fit(X_tr, y_tr)
    return tree

def evaluate_tree(tree, scaler, pca, env, feature_cols, *, episodes: int, max_steps: int = 1000, base_seed: int | None = None):
    """Evaluiert den Baum über *episodes* Episoden.
    Wenn *base_seed* übergeben wird, wird jeder Episoden‑Reset mit
    `env.reset(seed=base_seed + ep)` auf reproduzierbare Seeds gestellt.
    """
    rewards = []
    for ep in range(episodes):
        reset_out = env.reset(seed=None if base_seed is None else base_seed + ep)
        state = reset_out[0] if isinstance(reset_out, tuple) else reset_out
        total = 0.0
        for _ in range(max_steps):
            df_state = pd.DataFrame([state], columns=["x", "y", "vx", "vy", "theta", "v_theta", "left_leg", "right_leg"])
            state_scaled = scaler.transform(df_state[feature_cols])
            state_pca = pca.transform(state_scaled)
            action = int(tree.predict(state_pca)[0])
            step_out = env.step(np.clip(action, 0, env.action_space.n - 1))
            if len(step_out) == 5:
                next_state, reward, terminated, truncated, _ = step_out
                done = terminated or truncated
            else:
                next_state, reward, done, _ = step_out
            total += reward
            state = next_state
            if done:
                break
        rewards.append(total)
    return float(np.mean(rewards))

# ---------------------------------------------------------------------------
# ----------------------------- CLI Arguments -------------------------------
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="PCA + Decision‑Tree benchmark for LunarLander‑v2")

# Search‑phase parameters
parser.add_argument("--search_seeds", type=int, default=10, help="Number of random seeds in PCA search phase (default: 10)")
parser.add_argument("--episodes_per_seed", type=int, default=50, help="Episodes collected from PPO per search seed (default: 50)")
parser.add_argument("--search_eval_episodes", type=int, default=30, help="Episodes used to evaluate tree in search phase (default: 30)")

# Final phase parameters (training + evaluation)
parser.add_argument("--final_eval_seeds", type=int, default=10, help="Number of random seeds used solely for evaluation in the final phase (default: 10)")
parser.add_argument("--final_eval_episodes", type=int, default=100, help="Episodes per evaluation seed (default: 100)")
parser.add_argument("--max_depth", type=int, default=15, help="Maximum tree depth in final phase (depths 1..N, default: 15)")

# General
parser.add_argument("--model_path", type=str, default="models/ppo_LunarLander-v2/ppo-LunarLander-v2.zip", help="Path to PPO model")
parser.add_argument("--output_dir", type=str, default="pca_dt_runs", help="Base folder to save each run")
parser.add_argument("--top_k", type=int, default=5, help="Top‑k features from RandomForest importance (default: 5)")

ARGS = parser.parse_args()

# ---------------------------------------------------------------------------
# ------------------------------ Main logic ---------------------------------
# ---------------------------------------------------------------------------

def main():
    # 1) Prepare output folder (run_n)
    os.makedirs(ARGS.output_dir, exist_ok=True)
    run_idx = 1
    pat = re.compile(r"run_(\d+)")
    for folder in os.listdir(ARGS.output_dir):
        m = pat.fullmatch(folder)
        if m:
            run_idx = max(run_idx, int(m.group(1)) + 1)
    RUN_FOLDER = os.path.join(ARGS.output_dir, f"run_{run_idx}")
    os.makedirs(RUN_FOLDER, exist_ok=True)

    TREE_FOLDER = os.path.join(RUN_FOLDER, "decision_trees")
    os.makedirs(TREE_FOLDER, exist_ok=True)

    SEARCH_TREE_FOLDER = os.path.join(RUN_FOLDER, "search_phase_trees")
    os.makedirs(SEARCH_TREE_FOLDER, exist_ok=True)

    # Save CLI configuration
    with open(os.path.join(RUN_FOLDER, "configuration.json"), "w", encoding="utf-8") as f:
        json.dump(vars(ARGS), f, indent=2)

    # 2) Environment + PPO model
    env = gym.make("LunarLander-v2")
    agent = PPO.load(ARGS.model_path)

    # 3) Search phase – find best PCA seed
    search_seeds = random.sample(range(1_000_000), ARGS.search_seeds)
    best_seed: int | None = None
    best_reward = -np.inf
    best_bundle = None

    print("[Search] running", ARGS.search_seeds, "random seeds …")
    for idx, seed in enumerate(search_seeds, 1):
        records = collect_data(env, agent, episodes=ARGS.episodes_per_seed, seed=seed)
        X_raw, y = preprocess_data(records)
        X_sel, sel_cols = feature_selection(X_raw, y, k=ARGS.top_k)
        X_pca, scaler, pca = apply_pca(X_sel, variance=0.95, seed=seed)
        X_tr, _, y_tr, _ = train_test_split(X_pca, y, test_size=0.2, random_state=42)
        tree = train_tree(X_tr, y_tr, depth=3, seed=seed)
        reward = evaluate_tree(tree, scaler, pca, env, sel_cols, episodes=ARGS.search_eval_episodes)
        print(f"  {idx:3d}/{ARGS.search_seeds}: seed={seed:6d} mean_reward={reward:6.2f}")

        # Persist the search‑phase tree as TXT
        txt_path = os.path.join(SEARCH_TREE_FOLDER, f"search_tree_seed{seed}.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(export_text(tree))

        if reward > best_reward:
            best_reward = reward
            best_seed = seed
            best_bundle = dict(sel_cols=sel_cols, scaler=scaler, pca=pca, X_pca=X_pca, y=y)

    # 4) Save best seed details
    with open(os.path.join(RUN_FOLDER, "best_seed.txt"), "w", encoding="utf-8") as f:
        f.write(f"seed: {best_seed}\nmean_reward: {best_reward:.2f}\n")

    # 5) Final phase – train **einen** Baum pro Tiefe und evaluiere über mehrere Seeds
    evaluation_seeds = random.sample(range(1_000_000), ARGS.final_eval_seeds)
    aggregated_results: list[tuple[int, float]] = []  # (depth, mean_of_means)

    # Trainingsdaten basieren auf dem besten PCA‑Seed
    X_pca_best = best_bundle["X_pca"]
    y_best = best_bundle["y"]
    sel_cols = best_bundle["sel_cols"]
    scaler = best_bundle["scaler"]
    pca = best_bundle["pca"]

    X_train, _, y_train, _ = train_test_split(X_pca_best, y_best, test_size=0.2, random_state=42)

    depths = range(1, ARGS.max_depth + 1)
    print("\n[Final] training one tree per depth and evaluating across", ARGS.final_eval_seeds, "seeds …")
    for depth in depths:
        # -- train tree once (best_seed for reproducibility) --
        tree = train_tree(X_train, y_train, depth=depth, seed=best_seed)

        # Persist: joblib + txt
        joblib.dump(tree, os.path.join(TREE_FOLDER, f"tree_depth{depth}.joblib"))
        with open(os.path.join(TREE_FOLDER, f"tree_depth{depth}.txt"), "w", encoding="utf-8") as f:
            f.write(export_text(tree))

        # -- evaluate across many env seeds --
        mean_rewards: list[float] = []
        for ev_seed in evaluation_seeds:
            mean_r = evaluate_tree(
                tree,
                scaler,
                pca,
                env,
                sel_cols,
                episodes=ARGS.final_eval_episodes,
                base_seed=ev_seed,
            )
            mean_rewards.append(mean_r)
            print(f"  depth={depth:2d} eval_seed={ev_seed:6d} mean_reward={mean_r:6.2f}")

        depth_mean = float(np.mean(mean_rewards))
        aggregated_results.append((depth, depth_mean))
        print(f"→ depth={depth:2d} aggregated mean reward={depth_mean:6.2f}\n")

    env.close()

    # 6) Save aggregated results
    res_df = pd.DataFrame(aggregated_results, columns=["tree_depth", "mean_reward"])
    res_csv = os.path.join(RUN_FOLDER, "decision_tree_evaluation_final.csv")
    res_df.to_csv(res_csv, index=False)

    # 7) Plot
    plt.figure(figsize=(10, 6))
    plt.plot(res_df.tree_depth, res_df.mean_reward, marker="o")
    plt.title("Mean Reward vs. Tree Depth – aggregated across evaluation seeds")
    plt.xlabel("Tree depth")
    plt.ylabel("Mean reward (mean of seeds × episodes)")
    plt.grid(True)
    plt.savefig(os.path.join(RUN_FOLDER, "mean_reward_vs_depth.png"))
    print("Saved results to", RUN_FOLDER)


if __name__ == "__main__":
    main()
