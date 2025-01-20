import os
import gymnasium as gym
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from tqdm import tqdm

def collect_data(model_path, env_name='LunarLander-v3', n_episodes=500, max_steps=1000):
    """
    Sammle Zustands-Aktions-Paare aus dem vortrainierten PPO-Modell.

    Args:
        model_path (str): Pfad zum vortrainierten PPO-Modell.
        env_name (str): Name der Gym-Umgebung.
        n_episodes (int): Anzahl der Episoden zur Datensammlung.
        max_steps (int): Maximale Schritte pro Episode.

    Returns:
        pd.DataFrame: DataFrame mit Zuständen und Aktionen.
    """
    env = gym.make(env_name)
    model = PPO.load(model_path)

    data = []
    for episode in tqdm(range(n_episodes), desc="Daten sammeln"):
        # Entpacken der Rückgabe von env.reset()
        reset_output = env.reset()
        if isinstance(reset_output, tuple):
            obs, _ = reset_output
        else:
            obs = reset_output

        for step in range(max_steps):
            # Aktion vorhersagen
            action, _states = model.predict(obs, deterministic=True)
            data.append(np.concatenate((obs, [action])))

            # Schritt ausführen
            step_output = env.step(action)
            if len(step_output) == 5:
                obs, reward, terminated, truncated, info = step_output
                done = terminated or truncated
            else:
                # Fallback für ältere Gym-Versionen
                obs, reward, done, info = step_output

            if done:
                break

    env.close()
    # Angenommen, die LunarLander-Umgebung hat 8 Zustandsmerkmale
    columns = [
        'x_position', 'y_position',
        'x_velocity', 'y_velocity',
        'angle', 'angular_velocity',
        'left_leg_contact', 'right_leg_contact',
        'action'
    ]
    df = pd.DataFrame(data, columns=columns)
    return df

# Beispielaufruf (sicherstellen, dass der Pfad korrekt ist)
if __name__ == "__main__":
    # Pfad zu deinem vortrainierten PPO-Modell
    model_path = 'models/ppo-LunarLander-v3/best_model.zip'
    
    # Daten sammeln
    print("Starte Datensammlung...")
    df = collect_data(model_path)
    df.to_pickle('lunar_lander_data.pkl')
    print("Daten wurden in 'lunar_lander_data.pkl' gespeichert.")
    print(f"Gesammelte Daten: {df.shape}")
