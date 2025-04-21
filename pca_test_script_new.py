from __future__ import annotations

import argparse
import json
import os
import re
import random
import warnings

import gym
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text
from stable_baselines3 import PPO

SECURE_RNG = random.SystemRandom()

warnings.filterwarnings("ignore")


def collect_data(
    env,
    agent,
    *,
    num_samples: int,
    max_steps: int = 1000,
    seed: int | None = None,
):
    """
    Sammelt genau `num_samples` (state, action)-Tupel.
    Zeigt dabei einen tqdm‑Fortschrittsbalken.
    """
    data = []
    # erste Episode initialisieren
    reset_out = env.reset(seed=seed)
    state = reset_out[0] if isinstance(reset_out, tuple) else reset_out

    for _ in tqdm(range(num_samples), desc="Collecting Data", leave=False):
        action, _ = agent.predict(state, deterministic=True)
        data.append((state, action))

        step_out = env.step(int(action))
        if len(step_out) == 5:                # Gymnasium‑API
            next_state, _, terminated, truncated, _ = step_out
            done = terminated or truncated
        else:                                 # älteres Gym‑Fallback
            next_state, _, done, _ = step_out

        state = next_state
        if done:                              # Episode vorbei? → neu starten
            reset_out = env.reset(seed=None)
            state = reset_out[0] if isinstance(reset_out, tuple) else reset_out

    return data

def preprocess_data(records):
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

def evaluate_tree(
    tree,
    scaler,
    pca,
    env,
    feature_cols,
    *,
    episodes: int,
    max_steps: int = 1000,
    base_seed: int | None = None,
    test_seeds: list[int] | None = None,
) -> float:
    """
    Wenn test_seeds übergeben werden, ignoriert base_seed und nutzt
    genau diese Seeds zum Resetten der Umgebung.
    Ansonsten wie gehabt: base_seed + ep, oder seed=None.
    """
    rewards = []
    all_cols = ["x","y","vx","vy","theta","v_theta","left_leg","right_leg"]
    n_actions = env.action_space.n

    for ep in range(episodes):
        # 1) Wähle Reset‑Seed:
        if test_seeds is not None:
            seed = test_seeds[ep]
        elif base_seed is not None:
            seed = base_seed + ep
        else:
            seed = None

        reset_out = env.reset(seed=seed)
        state = reset_out[0] if isinstance(reset_out, tuple) else reset_out

        total_r = 0.0
        for _ in range(max_steps):
            # State → PCA‑Features
            row = pd.DataFrame([state], columns=all_cols)[feature_cols]
            scaled = scaler.transform(row)
            projected = pca.transform(scaled)
            a = int(tree.predict(projected)[0])
            a = np.clip(a, 0, n_actions - 1)

            step = env.step(a)
            # Gym/Gymnasium‑Fallback
            if len(step) == 5:
                next_state, reward, term, trunc, _ = step
                done = term or trunc
            else:
                next_state, reward, done, _ = step

            total_r += reward
            state = next_state
            if done:
                break

        rewards.append(total_r)

    return float(np.mean(rewards))


parser = argparse.ArgumentParser(description="PCA + Decision-Tree benchmark for LunarLander-v2")


parser.add_argument("--search_seeds", type=int, default=10, help="Number of random seeds in PCA search phase (default: 20)")
parser.add_argument("--num_samples",type=int,default=10000,help="Anzahl (state, action)-Paare pro PCA‑Seed (default: 10000)",)
parser.add_argument("--search_eval_episodes", type=int, default=10, help="Episodes used to evaluate tree in search phase (default: 30)")

parser.add_argument("--final_eval_seeds", type=int, default=5, help="Number of random seeds used solely for evaluation in the final phase (default: 10)")
parser.add_argument("--final_eval_episodes", type=int, default=10, help="Episodes per evaluation seed (default: 100)")
parser.add_argument("--max_depth", type=int, default=15, help="Maximum tree depth in final phase (depths 1..N, default: 15)")

parser.add_argument("--model_path", type=str, default="models/ppo_LunarLander-v2/ppo-LunarLander-v2.zip", help="Path to PPO model")
parser.add_argument("--output_dir", type=str, default="pca_dt_runs", help="Base folder to save each run")
parser.add_argument("--top_k", type=int, default=5, help="Top-k features from RandomForest importance (default: 5)")

ARGS = parser.parse_args()



def main():
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

    with open(os.path.join(RUN_FOLDER, "configuration.json"), "w", encoding="utf-8") as f:
        json.dump(vars(ARGS), f, indent=2)

    ENV_NAME = "LunarLander-v2"
    try:
        env = gym.make(ENV_NAME)
    except gym.error.Error as e:
        print(f"Error creating environment {ENV_NAME}: {e}")
        print("Please ensure you have 'pip install gym[box2d]' or 'pip install gymnasium[box2d]' installed.")
        return

    if not os.path.exists(ARGS.model_path):
        print(f"Error: PPO model not found at {ARGS.model_path}")
        print("Please ensure the path is correct and the model file exists.")
        env.close()
        return
    agent = PPO.load(ARGS.model_path)


    search_seeds = SECURE_RNG.sample(range(1_000_000), ARGS.search_seeds)
    best_seed: int | None = None
    best_reward = -np.inf
    best_bundle = None

    print(f"[Search] running {ARGS.search_seeds} random seeds to get the best PCA configuration in depth 3.")
    for idx, seed in enumerate(search_seeds, 1):
        records = collect_data(env,agent,num_samples=ARGS.num_samples,seed=seed,)
        if not records:
            print(f"Warning: No data collected for search seed {seed}. Skipping.")
            continue
        X_raw, y = preprocess_data(records)
        X_sel, sel_cols = feature_selection(X_raw, y, k=ARGS.top_k, rf_seed=seed)
        X_pca, scaler, pca = apply_pca(X_sel, variance=0.95, seed=seed)
        X_tr, _, y_tr, _ = train_test_split(X_pca, y, test_size=0.2, random_state=42)
        tree = train_tree(X_tr, y_tr, depth=3, seed=seed)
        test_seeds = SECURE_RNG.sample(range(1_000_000), ARGS.search_eval_episodes)
        reward = evaluate_tree(
            tree, scaler, pca, env, sel_cols,
            episodes=ARGS.search_eval_episodes,
            base_seed=None,         
            test_seeds=test_seeds,  
        )
        print(f"  {idx:3d}/{ARGS.search_seeds}: seed={seed:6d} mean_reward={reward:6.2f}")

        txt_path = os.path.join(SEARCH_TREE_FOLDER, f"search_tree_seed{seed}.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            pca_feature_names = [f"pca_comp_{i}" for i in range(X_pca.shape[1])]
            try:
                 f.write(export_text(tree, feature_names=pca_feature_names))
            except TypeError:
                 f.write(export_text(tree))


        if reward > best_reward:
            best_reward = reward
            best_seed = seed
            best_bundle = dict(sel_cols=sel_cols, scaler=scaler, pca=pca, X_pca=X_pca, y=y)

    if best_bundle is None or best_seed is None:
        print("Error: No suitable PCA configuration found in the search phase. Exiting.")
        env.close()
        return

    print(f"\nBest PCA seed found: {best_seed} with search reward: {best_reward:.2f}")
    with open(os.path.join(RUN_FOLDER, "best_seed_info.txt"), "w", encoding="utf-8") as f:
        f.write(f"Best seed found during search: {best_seed}\n")
        f.write(f"Mean reward during search evaluation (depth 3 tree, {ARGS.search_eval_episodes} episodes): {best_reward:.2f}\n")
        f.write("\nSelected features for PCA (top_k={}):\n".format(ARGS.top_k))
        for col in best_bundle["sel_cols"]:
            f.write(f"- {col}\n")

    scaler_path = os.path.join(RUN_FOLDER, "best_pca_scaler.joblib")
    pca_path = os.path.join(RUN_FOLDER, "best_pca_transformer.joblib")
    joblib.dump(best_bundle["scaler"], scaler_path)
    joblib.dump(best_bundle["pca"], pca_path)
    print(f"Saved best PCA scaler to {scaler_path}")
    print(f"Saved best PCA transformer to {pca_path}")

    selected_features_path = os.path.join(RUN_FOLDER, "selected_features.joblib")
    joblib.dump(best_bundle["sel_cols"], selected_features_path)
    print(f"Saved selected features list to {selected_features_path}")

    pca_explanation_path = os.path.join(RUN_FOLDER, "best_pca_components_explanation.txt")
    pca_obj = best_bundle["pca"]
    selected_features = best_bundle["sel_cols"]
    with open(pca_explanation_path, "w", encoding="utf-8") as f:
        f.write("PCA Component Explanation\n")
        f.write("=========================\n")
        f.write(f"Based on PCA seed: {best_seed}\n")
        f.write(f"Input features scaled before PCA: {', '.join(selected_features)}\n")
        f.write(f"Number of components selected (retaining >= 95% variance): {pca_obj.n_components_}\n\n")

        for i, component in enumerate(pca_obj.components_):
            f.write(f"Principal Component {i+1} (Explains {pca_obj.explained_variance_ratio_[i]:.2%} variance):\n")
            feature_weights = sorted(zip(selected_features, component), key=lambda x: abs(x[1]), reverse=True)
            for feature_name, weight in feature_weights:
                 f.write(f"  - {feature_name:<10}: {weight:+.4f}\n")
            f.write("\n")
    print(f"Saved PCA component explanation to {pca_explanation_path}")


    evaluation_seeds = SECURE_RNG.sample(range(1_000_000), ARGS.final_eval_seeds)
    aggregated_results: list[tuple[int, float, float]] = []

    X_pca_best = best_bundle["X_pca"]
    y_best = best_bundle["y"]
    sel_cols = best_bundle["sel_cols"]
    scaler = best_bundle["scaler"]
    pca = best_bundle["pca"]

    X_train, _, y_train, _ = train_test_split(X_pca_best, y_best, test_size=0.2, random_state=42)

    depths = list(range(1, ARGS.max_depth + 1))
    print(f"\n[Final] Training one tree per depth (1 to {ARGS.max_depth}) using best PCA config (seed {best_seed}).")
    print(f"Evaluating each tree across {ARGS.final_eval_seeds} distinct evaluation seeds ({ARGS.final_eval_episodes} episodes each)...")

    for depth in depths:
        tree = train_tree(X_train, y_train, depth=depth, seed=best_seed)

        joblib.dump(tree, os.path.join(TREE_FOLDER, f"tree_depth{depth}.joblib"))
        with open(os.path.join(TREE_FOLDER, f"tree_depth{depth}.txt"), "w", encoding="utf-8") as f:
             pca_feature_names = [f"pca_comp_{i}" for i in range(X_pca_best.shape[1])]
             try:
                 f.write(export_text(tree, feature_names=pca_feature_names))
             except TypeError:
                 f.write(export_text(tree))

        mean_rewards_for_this_depth: list[float] = []
        print(f"  Evaluating depth={depth:2d}:")
        for i, ev_seed in enumerate(evaluation_seeds, 1):
            mean_r = evaluate_tree(
                tree, scaler, pca, env, sel_cols,
                episodes=ARGS.final_eval_episodes,
                base_seed=ev_seed,    
                test_seeds=None
            )
            mean_rewards_for_this_depth.append(mean_r)
            print(f"    Eval seed {i:3d}/{ARGS.final_eval_seeds} ({ev_seed:6d}): mean_reward={mean_r:6.2f}")

        depth_mean = float(np.mean(mean_rewards_for_this_depth))
        depth_std = float(np.std(mean_rewards_for_this_depth))
        aggregated_results.append((depth, depth_mean, depth_std))
        print(f"  → Depth={depth:2d} aggregated: mean={depth_mean:6.2f} ± {depth_std:.2f} (std dev over {ARGS.final_eval_seeds} eval seeds)\n")

    env.close()

    res_df = pd.DataFrame(aggregated_results, columns=["tree_depth", "mean_reward", "std_reward"])
    res_csv = os.path.join(RUN_FOLDER, "decision_tree_evaluation_final.csv")
    res_df.to_csv(res_csv, index=False)
    print(f"Aggregated evaluation results saved to {res_csv}")

    plot_filename = os.path.join(RUN_FOLDER, "mean_reward_vs_depth.png")
    EXPERIMENT_NAME = f"PCA+DT Run {run_idx}"

    plt.figure(figsize=(12, 7))

    x = np.arange(len(depths))
    bar_width = 0.6

    rew = res_df['mean_reward'].values
    seeds_std = res_df['std_reward'].values

    plt.bar(
        x,
        rew,
        width=bar_width,
        yerr=seeds_std,
        capsize=4,
        label=f"{EXPERIMENT_NAME} (Error Bars: Std Dev over {ARGS.final_eval_seeds} Eval Seeds)",
        color="#1f77b4"
    )

    plt.xticks(x, depths)
    plt.xlabel("Tree Depth")
    plt.ylabel(f"Mean Reward (averaged over {ARGS.final_eval_seeds} seeds × {ARGS.final_eval_episodes} episodes)")
    plt.title(f"Decision Tree Performance vs. Max Depth ({ENV_NAME})\nExperiment: {EXPERIMENT_NAME}", fontsize=14)

    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.axhline(y=100, color='grey', linestyle=':', linewidth=1, label='Threshold 100')
    plt.axhline(y=200, color='darkgrey', linestyle=':', linewidth=1, label='Threshold 200')

    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_filename)
    print(f"Plot saved as '{plot_filename}'.")

    print(f"\nResults and artifacts saved to: {RUN_FOLDER}")


if __name__ == "__main__":
    main()
