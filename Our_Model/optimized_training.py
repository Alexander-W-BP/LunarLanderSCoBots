import os
import math
import torch
import gymnasium as gym

# Direkt die Klasse importieren, die das eigentliche Environment implementiert
from gymnasium.envs.box2d.lunar_lander import LunarLander
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback

class CustomLunarLanderEnv(LunarLander):
    def __init__(self, max_steps=1000, render_mode="rgb_array"):
        super().__init__(render_mode=render_mode)
        self._max_steps = max_steps
        self._current_step = 0

    def step(self, action):
        self._current_step += 1
        
        # (obs, reward, done, truncated, info) = super().step(action)
        obs, reward, done, truncated, info = super().step(action)

        # ----------------------------
        # 1) Euklidische Distanz: sqrt(x^2 + y^2), (obs[0] = x, obs[1] = y)
        distance_to_origin = math.sqrt(obs[0] ** 2 + obs[1] ** 2)
        
        # 2) Belohnung für Nähe: z.B. c / (1 + distance)
        #    Je kleiner die Distanz, desto größer das Summand
        c = 0.2
        distance_reward = c / (1.0 + distance_to_origin)
        reward += distance_reward
        
        if obs[6] == 1.0 or obs[7] == 1.0:
            reward += 20  # Belohnung für Landung

        return obs, reward, done, truncated, info

    def reset(self, seed=None, options=None):
        self._current_step = 0
        return super().reset(seed=seed, options=options)

def main():
    log_dir = "./logs/DQN_optimized"
    os.makedirs(log_dir, exist_ok=True)

    # Eigene Umgebung
    env = CustomLunarLanderEnv(max_steps=1000, render_mode="rgb_array")
    env = Monitor(env, log_dir)

    eval_callback = EvalCallback(
        env, 
        best_model_save_path=log_dir,
        log_path=log_dir, 
        eval_freq=2500, 
        deterministic=True, 
        render=False
    )

    policy_kwargs = dict(
        net_arch=[64, 64],
        activation_fn=torch.nn.ReLU
    )

    model = DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=1e-3,
        batch_size=64,
        buffer_size=100000,
        exploration_initial_eps=1.0,
        exploration_fraction=0.2,
        exploration_final_eps=0.01,
        gamma=0.99,
        tau=1.0,
        target_update_interval=1000,
        train_freq=(4, "step"),
        gradient_steps=1,
        max_grad_norm=10,
        policy_kwargs=policy_kwargs,
        verbose=1,
        seed=1
    )

    model.learn(
        total_timesteps=25000,
        callback=eval_callback
    )

    model.save(os.path.join(log_dir, "dqn_lunarlander_optimized"))
    print("Training beendet und Modell gespeichert.")

    # Optional: Evaluation
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
