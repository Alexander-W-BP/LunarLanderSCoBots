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

n_episodes and n_seeds are set to low values to make testing out the commands faster. We usually set both values to 100 to evaluate.

```bash
python3 dt_exp.py --experiment original_features --n_episodes 5 --n_seeds 5
python3 dt_exp.py --experiment pca_meta_features --n_episodes 5 --n_seeds 5
python3 dt_exp.py --experiment chat_gpt_features --n_episodes 5 --n_seeds 5
python3 dt_exp.py --experiment plots_features_full --n_episodes 5 --n_seeds 5
```

These commands create 'decision\*tree_experiments\*\*' folders. In such a folder you can find an evaluation run. Inside the run folder there is an image plotting the performance, a csv and a joblib file of different performance metrics and the results from the different tree_depths in a joblib file and a txt-file.

To evaluate the PCA-method, run the script pca_script.py. This script tries to get a high performing tree for depth 3 by creating several PCA-artefacts and choosing the best one for a full evaluation.

```bash
python3 pca_script.py --......
```

### 3. Visualize a model

To see a model in action you can use the following command:

```bash
python3 run_lander_agent.py <model_path>
```

An example model_path looks like this:

```bash
python3 run_lander_agent.py decision_tree_experiments_pca_meta_features/run_1/trees/good_tree_depth_10.joblib
```

It is important to specify the model*path relative to the root project because the folder 'decision_tree_experiments*\*' contains the experiment name which is important for the script to select the correct features for the model.
