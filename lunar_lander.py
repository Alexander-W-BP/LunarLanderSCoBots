# Imports
import io
import os
import glob
import torch
import base64

import numpy as np
import matplotlib.pyplot as plt

import stable_baselines3
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.results_plotter import ts2xy, load_results
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

import gymnasium as gym
from gym import spaces
import Box2D
from game_object import GameObjectInterface, LunarLanderObject


def get_new_log_dir(base_log_dir, algorithm, prefix="log_v"):
    """
    Erstellt einen neuen Log-Ordner für ein bestimmtes Algorithmus mit einem eindeutigen Namen.
    
    :param base_log_dir: Basisverzeichnis für Logs.
    :param algorithm: Name des Algorithmus (z.B. 'DQN', 'PPO').
    :param prefix: Präfix für den Log-Ordnernamen.
    :return: Pfad zum neu erstellten Log-Ordner.
    """
    # Erstelle den Pfad für den spezifischen Algorithmus
    algorithm_log_dir = os.path.join(base_log_dir, algorithm)
    os.makedirs(algorithm_log_dir, exist_ok=True)
    
    # Suche nach bestehenden Log-Verzeichnissen für den Algorithmus
    existing_dirs = glob.glob(os.path.join(algorithm_log_dir, f"{prefix}*"))
    version_numbers = []
    
    for dir_path in existing_dirs:
        basename = os.path.basename(dir_path)
        try:
            version = int(basename.replace(prefix, ""))
            version_numbers.append(version)
        except ValueError:
            continue  # Ignoriere Ordner, die dem Muster nicht entsprechen
    
    if version_numbers:
        new_version = max(version_numbers) + 1
    else:
        new_version = 1
    
    new_log_dir = os.path.join(algorithm_log_dir, f"{prefix}{new_version}")
    os.makedirs(new_log_dir, exist_ok=True)
    return new_log_dir

def extract_all_objects(env):
    """
    Extrahiert alle Objekte aus der Umgebung und deren Positionen.

    :param env: Die Gym-Umgebung.
    :return: Liste der extrahierten Objekte.
    """
    objects = []
    for attr_name in dir(env.unwrapped):
        if "drawlist" in attr_name:  # Überspringe drawlist-Attribute
            continue
        attr = getattr(env.unwrapped, attr_name, None)
        # Filtere physische Objekte (b2Body) und extrahiere Position
        if isinstance(attr, Box2D.b2Body):
            position = getattr(attr, 'position', (0, 0))
            objects.append(LunarLanderObject(name=attr_name, position=(position.x, position.y)))
        elif isinstance(attr, list):  # Beispiel: Beine oder Partikel
            for i, sub_attr in enumerate(attr, start=1):
                if isinstance(sub_attr, Box2D.b2Body):
                    position = getattr(sub_attr, 'position', (0, 0))
                    objects.append(LunarLanderObject(name=f"{attr_name}_{i}", position=(position.x, position.y)))
    return objects

def train_model(algorithm, env, log_dir, **kwargs):
    """
    Trainiert ein Modell basierend auf dem angegebenen Algorithmus.
    
    :param algorithm: Name des Algorithmus (z.B. 'DQN', 'PPO').
    :param env: Gym-Umgebung.
    :param log_dir: Verzeichnis zum Speichern der Logs.
    :param kwargs: Weitere Hyperparameter spezifisch für den Algorithmus, einschließlich 'policy_kwargs'.
    :return: Trainiertes Modell.
    """
    # Extrahiere policy_kwargs aus kwargs
    policy_kwargs = kwargs.pop('policy_kwargs', None)
    # Extrahiere total_timesteps und callback aus kwargs
    total_timesteps = kwargs.pop('total_timesteps', 50000)
    callback = kwargs.pop('callback', None)
    
    if algorithm == "DQN":
        model = DQN(
            "MlpPolicy",
            env,
            policy_kwargs=policy_kwargs,
            **kwargs  # Übergebe nur die verbleibenden kwargs an den Konstruktor
        )
    elif algorithm == "PPO":
        model = PPO(
            "MlpPolicy",
            env,
            policy_kwargs=policy_kwargs,
            **kwargs  # Übergebe nur die verbleibenden kwargs an den Konstruktor
        )
    else:
        raise ValueError(f"Algorithmus '{algorithm}' wird nicht unterstützt.")
    
    # Training des Modells
    model.learn(total_timesteps=total_timesteps, 
                log_interval=10, 
                callback=callback)
    print(f"{algorithm} Training abgeschlossen.")
    
    # Speichern des trainierten Modells
    model.save(os.path.join(log_dir, f"{algorithm.lower()}_lunar_lander"))
    print(f"Modell gespeichert in {os.path.join(log_dir, f'{algorithm.lower()}_lunar_lander.zip')}")
    
    return model

def evaluate_model(model, env_name, render=True):
    """
    Führt eine Episode mit dem gegebenen Modell aus und gibt die Gesamtbelohnung zurück.
    
    :param model: Trainiertes Modell.
    :param env_name: Name der Gym-Umgebung.
    :param render: Ob die Umgebung gerendert werden soll.
    :return: Gesamtbelohnung der Episode.
    """
    env = gym.make(env_name, render_mode="human" if render else "rgb_array")
    observation = env.reset()[0]
    total_reward = 0
    done = False
    while not done:
        if render:
            env.render()
        action, _states = model.predict(observation, deterministic=True)
        observation, reward, done, info, _ = env.step(action)
        total_reward += reward
    env.close()
    return total_reward

def plot_results(log_dirs, algorithms, title="Training Performance Comparison"):
    """
    Plottet die Trainingsbelohnungen für die verschiedenen Algorithmen.
    
    :param log_dirs: Liste der Log-Verzeichnisse für die Algorithmen.
    :param algorithms: Liste der Algorithmenamen.
    :param title: Titel des Plots.
    """
    plt.figure(figsize=(10, 6))
    
    for log_dir, algorithm in zip(log_dirs, algorithms):
        results = load_results(log_dir)
        x, y = ts2xy(results, 'timesteps')
        plt.plot(x, y, label=algorithm)
    
    plt.xlabel('Timesteps')
    plt.ylabel('Rewards')
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()

def main():
    # Basis-Log-Verzeichnis
    base_log_dir = "C:/Studium_TU_Darmstadt/Master/1. Semester/KI Praktikum/KI_Start/logs/"
    os.makedirs(base_log_dir, exist_ok=True)
    
    # Algorithmen, die trainiert werden sollen
    algorithms = ["DQN"]
    
    # Gemeinsame Policy-Parameter
    nn_layers = [64, 64]  # Zwei versteckte Schichten mit je 64 Neuronen
    policy_kwargs = dict(
        activation_fn=torch.nn.ReLU,
        net_arch=nn_layers
    )
    
    # Lernrate (kann je nach Algorithmus angepasst werden)
    learning_rates = {
        "DQN": 0.001,
        "PPO": 0.0003  # Standard-Lernrate für PPO
    }
    
    # Andere Hyperparameter (können je nach Algorithmus angepasst werden)
    hyperparams = {
        "DQN": {
            "batch_size": 64,  # Erhöht für bessere Lernfähigkeit
            "buffer_size": 10000,  # Erhöht für genügend Erfahrungsspeicher
            "learning_starts": 1000,  # Mehr Startschritte für effektiveres Lernen
            "gamma": 0.99,
            "tau": 1,
            "target_update_interval": 1000,  # Erhöht für stabilere Zielnetzwerke
            "train_freq": (1, "step"),
            "max_grad_norm": 10,
            "exploration_initial_eps": 1,
            "exploration_fraction": 0.1,  # Schnellere Reduzierung der Exploration
            "gradient_steps": 1,
            "seed": 1,
            "verbose": 0,
            # 'total_timesteps' und 'callback' werden separat übergeben
        },
        "PPO": {
            "n_steps": 2048,
            "batch_size": 64,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.01,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "seed": 1,
            "verbose": 0,
            # 'total_timesteps' und 'callback' werden separat übergeben
        }
    }
    
    # Umgebungsname
    env_name = 'LunarLander-v3'
    
    # Speichere die Log-Verzeichnisse für die spätere Vergleich
    log_dirs = []
    
    # Trainiere jeden Algorithmus
    trained_models = {}
    for algorithm in algorithms:
        print(f"\n=== Training {algorithm} ===")
        # Erstelle ein neues Log-Verzeichnis für den Algorithmus
        log_dir = get_new_log_dir(base_log_dir, algorithm, prefix="log_v")
        log_dirs.append(log_dir)
        print(f"Verwende Log-Verzeichnis: {log_dir}")
        
        # Erstelle die Gym-Umgebung
        env = gym.make(env_name, render_mode="rgb_array")
        print('State shape: ', env.observation_space.shape)
        print('Number of actions: ', env.action_space.n)
        
        # Extrahiere Objekte und füge sie der Umgebung hinzu
        extracted_objects = extract_all_objects(env)
        print(f"Extrahierte Objekte: {extracted_objects}")
        
        # Überwache die Umgebung und logge die Ergebnisse
        env = Monitor(env, log_dir)
        
        # EvalCallback zur periodischen Evaluierung des Agenten
        callback = EvalCallback(env, log_path=log_dir, deterministic=True, render=False)
        
        # Sammle die Hyperparameter für den aktuellen Algorithmus
        params = hyperparams[algorithm].copy()
        params['learning_rate'] = learning_rates[algorithm]
        params['callback'] = callback
        params['policy_kwargs'] = policy_kwargs
        params['total_timesteps'] = 50000  # Setze total_timesteps
        
        # Trainiere das Modell
        model = train_model(algorithm, env, log_dir, **params)
        trained_models[algorithm] = model
    
    # Vergleiche die Trainingsleistungen durch Plotten
    plot_results(log_dirs, algorithms, title="DQN vs PPO Training Performance")
    
    # Optional: Modelle evaluieren und die Gesamtbelohnung anzeigen
    print("\n=== Evaluierung der Modelle ===")
    for algorithm in algorithms:
        model = trained_models[algorithm]
        total_reward = evaluate_model(model, env_name, render=True)
        print(f"{algorithm} - Total Reward: {total_reward}")

if __name__ == "__main__":
    main()
