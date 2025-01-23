import os
import warnings
import numpy as np
from pathlib import Path

import gym
from sklearn.tree import DecisionTreeClassifier
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# VIPER-Import aus deinem utils/viper.py
# Passe diesen Importpfad ggf. an, falls du es anders abgelegt hast.
from utils.viper import VIPER

from joblib import load

warnings.filterwarnings("ignore")

EVAL_ENV_SEED = 84


# ------------------------------------------------------------------
# Hilfs-Klassen / -Funktionen
# ------------------------------------------------------------------

class SB3Model:
    """
    Wrapper-Klasse um ein SB3/PPO-Modell für die Nutzung in eval_agent.
    """
    def __init__(self, model):
        self.name = "Original SB3 Model"
        self.model = model

    def predict(self, obs, deterministic=True):
        # stable_baselines3-Modelle geben (action, state_info) zurück
        action, _ = self.model.predict(obs, deterministic=deterministic)
        return action, None


class DTClassifierModel:
    """
    Wrapper-Klasse um einen sklearn DecisionTreeClassifier für eval_agent.
    """
    def __init__(self, model):
        self.name = "DT Classifier Model"
        self.model = model

    def predict(self, obs, deterministic=True):
        # sklearn-DecisionsTrees geben ein numpy.array zurück
        out = self.model.predict(obs)
        return np.array(out), None


def eval_agent(model, env, episodes=10, obs_save_file=None, acts_save_file=None):
    """
    Spielt `episodes` Episoden in `env` mit dem angegebenen `model`.
    - Gibt den durchschnittlichen Reward (Mean Reward) zurück.
    - Speichert optional die Zustände und Aktionen in Dateien, wenn
      `obs_save_file` und `acts_save_file` angegeben sind.

    Erwartet:
      - model: Objekt mit Methode `predict(obs, deterministic=True)`
      - env: VecEnv / DummyVecEnv
    """
    current_episode = 0
    rewards = []
    steps = []

    obs_out_array = []
    acts_out_array = []

    obs = env.reset()
    current_rew = 0.0
    current_steps = 0

    while True:
        # Aktuelle Observation merken (DummyVecEnv -> obs ist shape (1, n_features))
        obs_out_array.append(obs[0])

        action, _ = model.predict(obs, deterministic=True)
        acts_out_array.append(action[0])

        obs, reward, done, info = env.step(action)
        current_rew += reward
        current_steps += 1

        if done:
            current_episode += 1
            # Bei VecEnv kann reward ein Array sein, also den ersten Eintrag nehmen
            if isinstance(current_rew, np.ndarray):
                current_rew = current_rew[0]
            rewards.append(current_rew)
            steps.append(current_steps)

            # Reset für nächste Episode
            current_rew = 0.0
            current_steps = 0
            obs = env.reset()

        if current_episode == episodes:
            mean_reward = np.mean(rewards)
            print("--------------------------------------------")
            print(model.name)
            print(f"Episoden Rewards: {rewards}")
            print(f"Mean Reward: {mean_reward:.2f}")
            print(f"Steps pro Episode: {steps} (Mean: {np.mean(steps):.2f})")
            print("--------------------------------------------\n")

            # Falls gewünscht: Speichere States/Actions
            if obs_save_file is not None:
                obs_save_file.unlink(missing_ok=True)
                np.save(obs_save_file, obs_out_array)
            if acts_save_file is not None:
                acts_save_file.unlink(missing_ok=True)
                np.save(acts_save_file, acts_out_array)

            return mean_reward


# ------------------------------------------------------------------
# Haupt-Skript
# ------------------------------------------------------------------

def main():
    # 1) PPO laden
    model_path = "models/ppo_LunarLander-v2/ppo-LunarLander-v2.zip"
    ppo_model = PPO.load(model_path)

    # 2) Gym-Umgebung erstellen & einmalig seed setzen
    env = gym.make('LunarLander-v2')
    obs = env.reset(seed=84)  # <--- Neu: Seeding über reset()

    # 3) DummyVecEnv ohne .seed()
    from stable_baselines3.common.vec_env import DummyVecEnv
    vec_env = DummyVecEnv([lambda: env])

    # 3) PPO eval + Daten sammeln
    #    - Wir spielen z.B. 50 Episoden mit dem PPO
    #    - Speichern die Observations/Actions in .npy-Files für VIPER
    episodes_to_collect = 50
    output_dir = Path("viper_output")
    output_dir.mkdir(parents=True, exist_ok=True)

    obs_file = output_dir / "obs.npy"
    acts_file = output_dir / "acts.npy"

    sb3_model = SB3Model(ppo_model)
    print("### Evaluierung des Original-PPO-Modells ###")
    eval_agent(sb3_model, vec_env, episodes=episodes_to_collect,
               obs_save_file=obs_file,
               acts_save_file=acts_file)

    # 4) VIPER-Konfiguration
    #    - Wir testen 3 verschiedene Tiefen
    #    - Du kannst NB_ITER und data_per_iter in VIPER anpassen
    depths = [3, 4, 5]
    NB_ITER = 10            # Anzahl VIPER-Iterationen
    DATA_PER_ITER = 15_000  # Datensamples pro Iteration (in viper.py 30_000 default)

    best_depth = None
    best_mean_reward = -999999.0
    best_tree_file = None

    # 5) Schleife über verschiedene max_depth-Werte
    for depth in depths:
        print(f"\n===== VIPER-Extraktion für max_depth={depth} =====")

        # Lege einen DecisionTreeClassifier mit gewünschter Tiefe an
        dtpolicy = DecisionTreeClassifier(max_depth=depth)

        # VIPER-Objekt: (Siehe deine viper.py)
        #   - model: Dein PPO
        #   - dtpolicy: Der DecisionTreeClassifier
        #   - env: Deine VecEnv
        #   - data_per_iter: Wieviele Daten-Samples pro Iteration gesammelt werden
        #   - rtpt: Wir haben keines, daher None
        vip = VIPER(model=ppo_model,
                    dtpolicy=dtpolicy,
                    env=vec_env,
                    rtpt=None,
                    data_per_iter=DATA_PER_ITER)

        # Starte VIPER (nb_iter = Anzahl der DAgger-ähnlichen Iterationen)
        vip.imitate(nb_iter=NB_ITER)

        # Speichere alle Trees + den besten Tree
        tree_output_dir = output_dir / f"viper_depth_{depth}"
        tree_output_dir.mkdir(exist_ok=True)
        vip.save_best_tree(tree_output_dir)

        # Hole den Namen der "_best.viper"-Datei
        all_viper_files = list(tree_output_dir.glob("*_best.viper"))
        if not all_viper_files:
            print("Keine _best.viper-Datei gefunden.")
            continue

        best_tree_of_depth = all_viper_files[0]  # Normalerweise nur eine
        print(f"Best Tree für Tiefe {depth}: {best_tree_of_depth.name}")

        # 6) Evaluierung des extrahierten DecisionTrees
        dtree_loaded = load(best_tree_of_depth)
        dt_wrapped_model = DTClassifierModel(dtree_loaded)

        # Teste z.B. 30 Episoden
        mean_reward = eval_agent(dt_wrapped_model, vec_env, episodes=30)

        # Prüfe, ob wir das beste Ergebnis haben
        if mean_reward > best_mean_reward:
            best_mean_reward = mean_reward
            best_depth = depth
            best_tree_file = best_tree_of_depth

    # 7) Ausgabe des besten Baumes
    if best_tree_file:
        print("====================================================")
        print(f"Bester Baum: Tiefe={best_depth}, Mean Reward={best_mean_reward:.2f}")
        print(f"Gefunden in: {best_tree_file}")
        print("====================================================")

        # Option: Umbenennen zu best_tree.viper im Hauptordner
        final_best_file = output_dir / "best_tree.viper"
        if final_best_file.exists():
            final_best_file.unlink()
        best_tree_file.rename(final_best_file)

        print(f"Best Tree wurde nach '{final_best_file}' verschoben.")
        print("Finaler Re-Eval:")
        final_dtree = load(final_best_file)
        final_model = DTClassifierModel(final_dtree)
        final_mean_reward = eval_agent(final_model, vec_env, episodes=30)
        print(f"Mean Reward des finalen Baumes: {final_mean_reward:.2f}\n")
    else:
        print("Kein best_tree.viper gefunden. Eventuell ist etwas schiefgelaufen.")


if __name__ == "__main__":
    main()
