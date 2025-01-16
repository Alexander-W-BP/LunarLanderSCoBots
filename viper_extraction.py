# Imports
import os
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text
from stable_baselines3 import PPO
import gymnasium as gym

# Pfad zu trainierten Modellen
MODEL_PATH = "models/ppo-LunarLander-v3/best_model.zip"

# Umgebungsname
ENV_NAME = 'LunarLander-v3'

# Hilfsfunktion: Lade das Modell
def load_model(model_path):
    """
    Lädt ein trainiertes PPO-Modell.
    """
    model = PPO.load(model_path)
    return model

# Funktion: VIPER-Extraktion durchführen
def viper_extraction(model_path, env_name, max_depth=5, num_samples=10000):
    print("\n=== VIPER-Extraktion für PPO ===")

    # Gym-Umgebung erstellen
    env = gym.make(env_name)

    # Modell laden
    model = load_model(model_path)

    # Daten sammeln: Beobachtungen und Aktionen
    observations = []
    actions = []
    
    obs = env.reset()[0]  # Erste Beobachtung
    for _ in range(num_samples):
        # Vorhersage der Aktion durch das Modell
        action, _ = model.predict(obs, deterministic=True)
        
        # Ursprüngliche Beobachtung
        observations.append(obs)
        actions.append(action)
        
        # Schritt in der Umgebung ausführen
        obs, _, done, _, _ = env.step(action)
        if done:
            obs = env.reset()[0]

    observations = np.array(observations)  # shape (num_samples, 8)
    actions = np.array(actions)            # shape (num_samples,)

    print(f"Gesammelte Daten: {observations.shape[0]} Beobachtungen "
          f"und {observations.shape[1]} Merkmale pro Beobachtung.")

    # Entscheidungsbaum trainieren
    clf = DecisionTreeClassifier(max_depth=max_depth, random_state=0)
    clf.fit(observations, actions)

    # Aktionsbeschreibungen
    action_descriptions = [
        "0: No Action",
        "1: Fire Left Booster",
        "2: Fire Main Booster",
        "3: Fire Right Booster"
    ]

    # Entscheidungsbaum ausgeben
    tree_rules = export_text(
        clf,
        feature_names=[
            "x_space",          # obs[0]
            "y_space",          # obs[1]
            "vel_x_space",      # obs[2]
            "vel_y_space",      # obs[3]
            "angle",            # obs[4]
            "angular_vel",      # obs[5]
            "leg_1",            # obs[6]
            "leg_2"             # obs[7]
        ]
    )

    # Aktionsbeschreibungen hinzufügen
    for class_value, description in enumerate(action_descriptions):
        tree_rules = tree_rules.replace(f"class: {class_value}", f"class: {description}")

    print("\nEntscheidungsbaum-Regeln mit Aktionsbeschreibungen:\n")
    print(tree_rules)

    # Entscheidungsbaum in eine Datei speichern
    output_file_path = "decision_tree.txt"
    with open(output_file_path, "w") as file:
        file.write(tree_rules)

    print(f"Entscheidungsbaum gespeichert unter: {output_file_path}")

    # Rückgabe des Entscheidungsbaums
    return clf

# Hauptprogramm
def main():
    try:
        clf = viper_extraction(
            model_path=MODEL_PATH,
            env_name=ENV_NAME,
            max_depth=3,  # Maximale Tiefe des Entscheidungsbaums
            num_samples=10000  # Anzahl der gesammelten Datenpunkte
        )
        print("\nExtraktion abgeschlossen.")
    except Exception as e:
        print(f"Fehler bei der VIPER-Extraktion: {e}")

if __name__ == "__main__":
    main()
