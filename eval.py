# Imports
import os
import numpy as np
from stable_baselines3 import DQN, PPO
import gymnasium as gym

def load_model(algorithm, model_dir):
    """
    Lädt ein trainiertes Modell basierend auf dem Algorithmus.

    :param algorithm: Name des Algorithmus (z.B. 'DQN', 'PPO').
    :param model_dir: Pfad zum Modell.
    :return: Geladenes Modell.
    """
    if algorithm == "DQN":
        model = DQN.load(os.path.join(model_dir, "dqn_lunar_lander"))
    elif algorithm == "PPO":
        model = PPO.load(os.path.join(model_dir, "ppo_lunar_lander"))
    else:
        raise ValueError(f"Algorithmus '{algorithm}' wird nicht unterstützt.")
    return model

def evaluate_agent(model, env_name, num_episodes=10, render=False):
    """
    Bewertet einen Agenten über mehrere Episoden.

    :param model: Trainiertes Modell.
    :param env_name: Name der Gym-Umgebung.
    :param num_episodes: Anzahl der Episoden zur Bewertung.
    :param render: Ob die Umgebung gerendert werden soll.
    :return: Durchschnittliche Belohnung über die Episoden.
    """
    env = gym.make(env_name, render_mode="human" if render else None)
    total_rewards = []

    for episode in range(num_episodes):
        obs = env.reset()[0]
        done = False
        episode_reward = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, _ = env.step(action)
            episode_reward += reward
            if render:
                env.render()

        total_rewards.append(episode_reward)
        print(f"Episode {episode + 1}: Belohnung = {episode_reward}")

    env.close()
    average_reward = np.mean(total_rewards)
    print(f"\nDurchschnittliche Belohnung über {num_episodes} Episoden: {average_reward}")
    return average_reward

# Hauptprogramm
def main():
    # Einstellungen
    algorithm = "DQN"  # Oder "PPO"
    model_dir = "./logs/DQN/log_v4"
    env_name = "LunarLander-v3"
    num_episodes = 10
    render = False  # Setzen Sie auf False, um die Visualisierung zu deaktivieren

    # Modell laden
    print(f"Lade {algorithm}-Modell...")
    model = load_model(algorithm, model_dir)

    # Agent evaluieren
    print(f"Bewerte {algorithm}-Agenten...")
    average_reward = evaluate_agent(model, env_name, num_episodes, render)
    print(f"Durchschnittliche Belohnung: {average_reward}")

if __name__ == "__main__":
    main()
