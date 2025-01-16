# Imports
import os
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text
from stable_baselines3 import DQN, PPO
import gymnasium as gym
import dtcontrol

# Pfad zu trainierten Modellen
MODEL_DIR = "C:/Studium_TU_Darmstadt/Master/1. Semester/KI Praktikum/KI_Start/logs/"

# Umgebungsname
ENV_NAME = 'LunarLander-v3'

# Hilfsfunktion: Lade das Modell basierend auf dem Algorithmus
def load_model(algorithm, model_dir):
    if algorithm == "DQN":
        model = DQN.load(os.path.join(model_dir, "dqn_lunar_lander"))
    elif algorithm == "PPO":
        model = PPO.load(os.path.join(model_dir, "ppo_lunar_lander"))
    else:
        raise ValueError(f"Algorithmus '{algorithm}' wird nicht unterstützt.")
    return model

# Funktion: VIPER-Extraktion durchführen
def viper_extraction(algorithm, model_dir, env_name, max_depth=5, num_samples=10000):
    print(f"\n=== VIPER-Extraktion für {algorithm} ===")
    
    # Gym-Umgebung erstellen
    env = gym.make(env_name, render_mode=None)

    # Modell laden
    model = load_model(algorithm, model_dir)

    # Daten sammeln: Beobachtungen und Aktionen
    observations = []
    actions = []
    
    obs = env.reset()[0]
    for _ in range(num_samples):
        # Vorhersage der Aktion durch das RL-Modell
        action, _ = model.predict(obs, deterministic=True)
        
        # -----------------------------------------------------------
        # NEUES FEATURE: ed_lander_moon = |vertical_position| = |obs[1]|
        ed_lander_moon = abs(obs[1])
        
        # Erweitere den ursprünglichen Beobachtungsvektor um 1 Dimension:
        extended_obs = np.concatenate([obs, [ed_lander_moon]])
        # -----------------------------------------------------------
        
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
    # ACHTUNG: Jetzt haben wir 9 Feature-Namen (die neue 9. heißt "ed_lander_moon")
    tree_rules = export_text(
        clf,
        feature_names=[
            "horizontal_position",  # obs[0]
            "vertical_position",    # obs[1]
            "horizontal_velocity",  # obs[2]
            "vertical_velocity",    # obs[3]
            "angle",               # obs[4]
            "angular_velocity",     # obs[5]
            "left_leg_contact",     # obs[6]
            "right_leg_contact",    # obs[7]
            "ed_lander_moon"        # obs[8] (neu)
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

    # Optional: VIPER-Algorithmus anwenden (kommentiert)
    """
    viper = Viper(env, model, max_iterations=10, epsilon=0.1, max_depth=max_depth)
    viper_tree = viper.fit()
    print("\nVIPER Entscheidungsbaum:\n")
    print(viper_tree)
    return clf, viper_tree
    """

    # Damit das Skript in main() nicht crasht, geben wir 2 Werte zurück, 
    # obwohl wir viper_tree weglassen:
    return clf, None

# Hauptprogramm: Entscheide, welches Modell extrahiert wird
def main():
    algorithms = ["DQN"]  # z. B. nur DQN
    
    for algorithm in algorithms:
        model_dir = os.path.join(MODEL_DIR, algorithm, "log_v4")  # Passe den Pfad ggf. an
        try:
            clf, viper_tree = viper_extraction(algorithm, model_dir, ENV_NAME, max_depth=5)
            print(f"\nExtraktion für {algorithm} abgeschlossen.\n")
        except Exception as e:
            print(f"Fehler bei {algorithm}: {e}")

if __name__ == "__main__":
    main()
