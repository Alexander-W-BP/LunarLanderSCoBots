# 🚀 Installation Process

We recommend using **conda 23.11.0** or newer to set up the environment.

## 📥 Create the Conda Environment

Run this command to install the dependencies:

```bash
conda env create -f environment.yml
```

## 🎯 Activate the Environment
Once installed, activate the transparent-ai environment with:

```bash
conda activate transparent-ai
```

# Commands

```bash
python train_experiment.py --experiment original_features --n_episodes 100 --n_seeds 100
python train_experiment.py --experiment pca_features --n_episodes 100 --n_seeds 100
python train_experiment.py --experiment chat_gpt_features --n_episodes 100 --n_seeds 100
python train_experiment.py --experiment top_5_features_only --n_episodes 100 --n_seeds 100
python train_experiment.py --experiment all_features --n_episodes 100 --n_seeds 100
python train_experiment.py --experiment original_pca_features --n_episodes 100 --n_seeds 100
python train_experiment.py --experiment plots_features_full --n_episodes 100 --n_seeds 100
python train_experiment.py --experiment plots_features_only --n_episodes 100 --n_seeds 100
```