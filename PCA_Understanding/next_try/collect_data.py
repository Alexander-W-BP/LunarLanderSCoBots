"""
Datei: collect_data.py

Beschreibung:
-------------
- Lädt das vortrainierte PPO-Modell (best_model.zip) für LunarLander-v3 (Gymnasium).
- Erzeugt Trajektorien-Daten (Zustand, Aktion) über num_episodes Episoden.
- Speichert die Daten als .npz-Datei.
- Zusätzlich werden erste statistische Infos zu den Rewards ausgegeben.
"""

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from tqdm import tqdm  # Import von tqdm für den Ladebalken

def collect_data(env_id="LunarLander-v3",
                model_path="models/ppo-LunarLander-v3/best_model.zip",
                num_episodes=1000,
                output_file="expert_data.npz"):
    # Umgebung erstellen
    env = gym.make(env_id, render_mode=None)
    model = PPO.load(model_path)
    
    states = []
    actions = []
    episode_rewards = []

    # Verwendung von tqdm für die Episoden
    for episode in tqdm(range(1, num_episodes + 1), desc="Episoden sammeln", unit="Episode"):
        # Umgebung zurücksetzen und entpacken der Rückgabe
        obs, info = env.reset()
        done = False
        total_reward = 0.0
        
        while not done:
            # Vorhersage des Modells
            action, _states = model.predict(obs, deterministic=True)
            states.append(obs)
            actions.append(action)
            
            # Aktion in der Umgebung ausführen
            step_result = env.step(action)
            
            # Entpacken der Rückgabe von step()
            if len(step_result) == 5:
                obs, reward, done, truncated, info = step_result
                done = done or truncated
            else:
                # Für den Fall, dass die Umgebung anders zurückgibt
                obs, reward, done, info = step_result
            
            total_reward += reward
        
        episode_rewards.append(total_reward)
    
    env.close()

    # Daten speichern
    np.savez(output_file, states=np.array(states), actions=np.array(actions))
    print(f"\n[collect_data] Expert data gespeichert unter '{output_file}'.")
    print(f"[collect_data] Gesammelte Episoden: {num_episodes}")
    print(f"[collect_data] Ø Reward des PPO-Modells in der Sammlung: {np.mean(episode_rewards):.2f}")

if __name__ == "__main__":
    collect_data()
