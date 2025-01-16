import os
import torch
import gymnasium as gym

# Direkt die Klasse importieren, die das eigentliche Environment implementiert
from gymnasium.envs.box2d.lunar_lander import LunarLander
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback


class CustomLunarLanderEnv(LunarLander):
    def __init__(self, max_steps=1000, render_mode="rgb_array"):
        # Aufruf des Original-Konstruktors von LunarLander
        super().__init__(render_mode=render_mode)
        self._max_steps = max_steps
        self._current_step = 0

    def step(self, action):
        self._current_step += 1
        # Gymnasium gibt (obs, reward, done, truncated, info) zurück
        obs, reward, done, truncated, info = super().step(action)

        # Bestrafe pro Zeitschritt minimal
        reward -= 0.1

        

        # Schließe Episode ab, wenn zu viele Schritte vergangen sind:
        # Wir setzen hier "truncated = True" (im Sinne von Early Stopping).
        if self._current_step >= self._max_steps:
            truncated = True
            reward -= 25  # Straf-Reward für zu langes Warten

        return obs, reward, done, truncated, info

    def reset(self, seed=None, options=None):
        self._current_step = 0
        return super().reset(seed=seed, options=options)


def main():
    # Neues Log-Verzeichnis erstellen
    log_dir = "./logs/DQN_optimized"
    os.makedirs(log_dir, exist_ok=True)

    # Custom-Umgebung instanziieren
    # Wichtig: Wir rufen NICHT gym.make(), sondern erzeugen das Environment direkt
    env = CustomLunarLanderEnv(max_steps=1000, render_mode="rgb_array")

    # Falls du Monitor-Statistiken loggen möchtest
    env = Monitor(env, log_dir)

    # EvalCallback mit derselben Umgebung (oder einer frischen Instanz)
    eval_callback = EvalCallback(
        env, 
        best_model_save_path=log_dir,
        log_path=log_dir, 
        eval_freq=5000, 
        deterministic=True, 
        render=False
    )

    # Netzwerkarchitektur
    policy_kwargs = dict(
        net_arch=[64, 64],
        activation_fn=torch.nn.ReLU
    )

    # DQN Hyperparameter
    model = DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=1e-3,          # ggf. anpassen
        batch_size=64,
        buffer_size=100000,          # größerer Replay Buffer
        exploration_initial_eps=1.0,
        exploration_fraction=0.2,    # längere Explorationsphase
        exploration_final_eps=0.01,
        gamma=0.99,
        tau=1.0,
        target_update_interval=1000,
        train_freq=(4, "step"),      # öfter trainieren
        gradient_steps=1,
        max_grad_norm=10,
        policy_kwargs=policy_kwargs,
        verbose=1,
        seed=1
    )

    # Training
    model.learn(
        total_timesteps=50000,  # mehr Trainingsschritte
        callback=eval_callback
    )

    # Speichern
    model.save(os.path.join(log_dir, "dqn_lunarlander_optimized"))
    print("Training beendet und Modell gespeichert.")

    # Optionale Evaluierung (mit 'human'-Render, wenn du möchtest)
    test_env = gym.make("LunarLander-v3", render_mode="human")
    obs, _ = test_env.reset()
    done = False
    truncated = False
    total_reward = 0
    while not (done or truncated):
        action, _ = model.predict(obs)
        obs, reward, done, truncated, info = test_env.step(action)
        total_reward += reward
        test_env.render()
    test_env.close()
    print("Evaluierungs-Reward:", total_reward)


if __name__ == "__main__":
    main()
