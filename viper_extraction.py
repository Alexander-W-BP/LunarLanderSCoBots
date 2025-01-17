# viper_extraction_compare_depths.py

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from stable_baselines3 import PPO
import gymnasium as gym

# Pfad zu trainierten Modellen
MODEL_PATH = "models/ppo-LunarLander-v3/best_model.zip"
ENV_NAME = 'LunarLander-v3'

def load_model(model_path):
    """Lädt ein trainiertes PPO-Modell."""
    model = PPO.load(model_path)
    return model

def main(num_samples=10000):
    print("\n=== VIPER-Extraktion und Vergleich verschiedener max_depth ===")

    # Schritt 1: Daten EINMAL sammeln
    env = gym.make(ENV_NAME)
    model = load_model(MODEL_PATH)

    observations = []
    actions = []

    obs = env.reset()[0]
    for _ in range(num_samples):
        action, _ = model.predict(obs, deterministic=True)
        observations.append(obs)
        actions.append(action)
        obs, _, done, _, _ = env.step(action)
        if done:
            obs = env.reset()[0]

    observations = np.array(observations)  # shape (num_samples, 8)
    actions = np.array(actions)            # shape (num_samples,)

    print(f"Gesammelte Daten: {observations.shape[0]} Beobachtungen, "
          f"{observations.shape[1]} Merkmale pro Beobachtung.")

    # Schritt 2: Train/Test-Split
    X_train, X_test, y_train, y_test = train_test_split(
        observations, 
        actions, 
        test_size=0.2, 
        random_state=42
    )

    # Schritt 3: Für jeden max_depth von 1..30 trainieren und Accuracy messen
    depths = range(1, 31)  # bis einschließlich 30
    accuracy_list = []

    for depth in depths:
        clf = DecisionTreeClassifier(max_depth=depth, random_state=0)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        accuracy_list.append(acc)
        print(f"max_depth={depth}: Test-Accuracy = {acc:.3f}")

    # Schritt 4: Grafische Ausgabe
    plt.figure(figsize=(8, 6))
    plt.plot(depths, accuracy_list, marker='o')
    plt.xlabel("Max Depth")
    plt.ylabel("Test Accuracy")
    plt.title("Decision Tree Accuracy vs. Max Depth (LunarLander-v3)")
    plt.grid(True)
    plt.savefig("accuracy_vs_depth.png")
    plt.show()

    print("Die Grafik wurde als 'accuracy_vs_depth.png' gespeichert.")

if __name__ == "__main__":
    main()
