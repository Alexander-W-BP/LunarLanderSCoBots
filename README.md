---
library_name: stable-baselines3
tags:
- LunarLander-v3
- deep-reinforcement-learning
- reinforcement-learning
- stable-baselines3
model-index:
- name: PPO
  results:
  - task:
      type: reinforcement-learning
      name: reinforcement-learning
    dataset:
      name: LunarLander-v3
      type: LunarLander-v3
    metrics:
    - type: mean_reward
      value: 279.56 +/- 18.47
      name: mean_reward
      verified: false
---

# **PPO** Agent playing **LunarLander-v3**
This is a trained model of a **PPO** agent playing **LunarLander-v3**
using the [stable-baselines3 library](https://github.com/DLR-RM/stable-baselines3).

## Usage (with Stable-baselines3)

```python
from stable_baselines3 import PPO
from huggingface_sb3 import load_from_hub

model_filename = load_from_hub(your_repo_id, your_filename)

model = PPO.load(model_filename)

env = gym.make({env_id}, render_mode="human")
    obs, info = env.reset()

    done = False

    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, done, truncated, info = env.step(action)
        env.render()

    env.close()
```
