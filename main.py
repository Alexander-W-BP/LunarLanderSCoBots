from stable_baselines3 import PPO
import gymnasium as gym

# Pfad zur gespeicherten Modell-Datei
model_path = "models/ppo-LunarLander-v3/best_model.zip"

# Lade das Modell
model = PPO.load(model_path)

# LunarLander-v3-Umgebung erstellen
env = gym.make("LunarLander-v3", render_mode="human")

# Simulation starten
obs, info = env.reset()
done = False

while not done:
    # Aktion vom Modell vorhersagen
    action, _states = model.predict(obs, deterministic=True)
    # Aktion in der Umgebung ausführen
    obs, rewards, done, truncated, info = env.step(action)
    # Umgebung rendern
    env.render()

# Umgebung schließen
env.close()
