#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gym
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.tree import export_text, plot_tree
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier

def main():
    # -----------------------------
    # 1) Laden des gespeicherten Baums und Preprocessing-Artefakte
    # -----------------------------
    # Passen die Dateinamen an deinen Speicherort an!
    tree_path = "xgb_policy.joblib"       # Dein trainiertes Modell
    scaler_path = "scaler.joblib"            # Dein Skalierer (falls genutzt)
    pca_path = "pca.joblib"                  # Dein PCA-Modell (falls genutzt)
    feat_path = "selected_features.joblib"   # Liste der ausgewählten Feature-Namen
    
    print("Lade Decision Tree ...")
    tree_model = joblib.load(tree_path)  # z.B. DecisionTreeClassifier
    
    # Prüfen, ob dein Modell tatsächlich ein DecisionTreeClassifier ist
    if not isinstance(tree_model, DecisionTreeClassifier):
        print("Warnung: Das geladene Modell ist kein DecisionTreeClassifier!")
    
    print("Lade Preprocessing-Artefakte ...")
    scaler = joblib.load(scaler_path)    # StandardScaler
    pca = joblib.load(pca_path)          # PCA
    selected_features = joblib.load(feat_path)  # z. B. ['x','y','vx','vy']
    print(f"Selektierte Features: {selected_features}")

    # -----------------------------
    # 2) Baum als Text ausgeben
    # -----------------------------
    print("\n=== Entscheidungsbaum (Textform) ===")
    # Achtung: Falls PCA genutzt wurde, hat dein Tree "n_features_in_" = #PCA-Komponenten
    # Die originalen Feature-Namen entsprechen also nicht 1:1 dem Input des Baums.
    # Man kann symbolisch "PC0", "PC1", ... nehmen:
    feature_names_for_print = [f"PC{i}" for i in range(tree_model.n_features_in_)]
    
    tree_text = export_text(tree_model, feature_names=feature_names_for_print)
    print(tree_text)

    # -----------------------------
    # 3) Baum als Grafik plotten
    # -----------------------------
    # Falls du nur Text möchtest, kannst du diesen Schritt überspringen
    print("Plotte Entscheidungsbaum ...")
    plt.figure(figsize=(12,8))
    plot_tree(
        decision_tree=tree_model,
        feature_names=feature_names_for_print,
        filled=True,
        proportion=True
    )
    plt.title("Entscheidungsbaum (Tiefe = {})".format(tree_model.get_depth()))
    plt.show()

    # -----------------------------
    # 4) Demo-Episode in LunarLander
    # -----------------------------
    # Achtung: In neueren Gym-Versionen (>= 0.26) => env = gym.make("LunarLander-v2", render_mode="human")
    print("\nStarte Demo-Episode im LunarLander ...")
    env = gym.make("LunarLander-v2", render_mode="human")
    
    num_episodes = 2
    max_steps = 600
    
    for ep in range(num_episodes):
        obs = env.reset()
        if isinstance(obs, tuple):
            obs, _info = obs  # neuere Gym-Verhaltensweise
        total_reward = 0
        
        for step in range(max_steps):
            # Vorverarbeitung: 
            # 1) in DataFrame packen => nur ausgewählte Features
            # 2) scaler.transform, pca.transform
            state_df = pd.DataFrame([obs], columns=["x","y","vx","vy","theta","v_theta","left_leg","right_leg"])
            state_sel = state_df[selected_features]
            state_scaled = scaler.transform(state_sel)
            state_pca = pca.transform(state_scaled)
            
            # Baum-Aktion vorhersagen
            action_pred = tree_model.predict(state_pca)[0]
            action_pred = int(action_pred)
            
            # Step in der Env
            s_out = env.step(action_pred)
            if len(s_out) == 5:
                # Gym v.26 => (next_obs, reward, terminated, truncated, info)
                next_obs, rew, terminated, truncated, info = s_out
                done = terminated or truncated
            else:
                # ältere Gym-Versionen => (next_obs, reward, done, info)
                next_obs, rew, done, info = s_out
            obs = next_obs
            total_reward += rew
            
            # Gym rendert (in neueren Versionen erfolgt das bei step())
            # Falls du eine ältere Version hast, ggf. env.render()
            
            if done:
                print(f"Episode {ep+1}, Schritt {step}, Reward = {total_reward:.2f}")
                break
    
    env.close()
    print("Demo beendet.")

if __name__ == "__main__":
    main()
