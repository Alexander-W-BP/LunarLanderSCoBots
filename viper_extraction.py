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
        
        # NEUES FEATURE: Abstand zur Zielplattform
        distance_to_target = np.sqrt(obs[0]**2 + obs[1]**2)
        
        # Erweiterte Beobachtung mit relationalem Feature
        extended_obs = np.concatenate([obs, [distance_to_target]])
        
        observations.append(extended_obs)
        actions.append(action)
        
        # Schritt in der Umgebung ausführen
        obs, _, done, _, _ = env.step(action)
        if done:
            obs = env.reset()[0]

    observations = np.array(observations)  # shape (num_samples, 9)
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
            "horizontal_position",  # obs[0]
            "vertical_position",    # obs[1]
            "horizontal_velocity",  # obs[2]
            "vertical_velocity",    # obs[3]
            "angle",                # obs[4]
            "angular_velocity",     # obs[5]
            "left_leg_contact",     # obs[6]
            "right_leg_contact",    # obs[7]
            "distance_to_target"    # obs[8] (neu)
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
            max_depth=5,  # Maximale Tiefe des Entscheidungsbaums
            num_samples=10000  # Anzahl der gesammelten Datenpunkte
        )
        print("\nExtraktion abgeschlossen.")
    except Exception as e:
        print(f"Fehler bei der VIPER-Extraktion: {e}")

if __name__ == "__main__":
    main()
