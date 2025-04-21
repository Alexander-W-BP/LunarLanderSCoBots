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

To evaluate the PCA-method, run the script pca_script.py. It performs a systematic search to determine the optimal PCA configuration for depth 3 by evaluating several randomly selected seeds. We have described in the paper that we ourselves have not yet fully understood why this method is so inconsistent. However, in order to create a reliable tree of depth 3 that also has very good performance, a suitable PCA configuration must be found. This is achieved by phase 1, where the best PCA configuration is used at the end. The number of seeds is therefore set to 20 by default.

All default values can be viewed in the file. These can of course be adjusted via the corresponding parameters. Example calls look like this:

```bash
python3 pca_script.py
python3 pca_script.py --final_eval_seeds 15 --final_eval_episodes 100 --max_depth 15 --top_k 6
```

If no result with a positive mean reward was found in the search phase, run the script again if you want to see a really good depth 3 tree.

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
