# Interpretable Reinforcement Learning in the Lunar Lander Environment

Project Overview...

## Installation

### 1. System packages

```bash
sudo apt-get update
sudo apt-get install swig
sudo apt-get install gcc-c++
sudo apt-get install python3-devel
sudo apt-get install -y python3-venv
```


### 2. Setup Project

```bash
python3 -m venv env
sudo apt install python3-pip
pip install -U pip && pip install -r requirements.txt
```

## Features

### 1. Get Action Space Division Plots:
```bash
python3 action_space_division_plots.py
```

### 2. Evaluate Feature Combinations:
- n_episodes and n_seeds are set to low values to make testing out the commands faster. We usually set both values to 100 to evaluate.

```bash
python dt_exp.py --experiment original_features --n_episodes 5 --n_seeds 5
python dt_exp.py --experiment pca_meta_features --n_episodes 5 --n_seeds 5
python dt_exp.py --experiment chat_gpt_features --n_episodes 5 --n_seeds 5
python dt_exp.py --experiment plots_features_full --n_episodes 5 --n_seeds 5
```
