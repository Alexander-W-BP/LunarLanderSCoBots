import gym
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import joblib
import os

from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Neu hinzugefügt: Random Forest für Feature Selection
from sklearn.ensemble import RandomForestClassifier

# OPTIONAL: Wenn du den PPO-Agenten nutzt
from stable_baselines3 import PPO

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------
# -------------------------- EINSTELLUNGEN / PARAMETER ------------------------
# ----------------------------------------------------------------------------
CONFIG = {
    # Name der Gym-Umgebung
    'ENV_NAME': 'LunarLander-v2',
    
    # Falls du einen PPO-Agenten hast, hier Pfad angeben
    'PPO_MODEL_PATH': 'models/ppo_LunarLander-v2/ppo-LunarLander-v2.zip',
    
    # Anzahl Episoden und Schritte zum Datensammeln
    'NUM_EPISODES_COLLECT': 50,
    'MAX_STEPS_COLLECT': 1000,
    
    # Anzahl der Komponenten für PCA (auf den ausgewählten Top-Features)
    'N_PCA_COMPONENTS': 4,
    
    # Tiefe der Bäume, die wir trainieren wollen
    'TREE_DEPTHS': [1, 2, 3, 4, 5, 6, 7],
    'NUM_TREES_PER_DEPTH': 3,
    
    # Split-Größe für Training/Test
    'TEST_SIZE': 0.2,
    'RANDOM_STATE': 1,
    
    # Wie viele Features sollen nach Random-Forest-Selection genutzt werden?
    'TOP_N_FEATURES': 5,
    
    # Ordner, in dem alle neu erstellten Artefakte und Ergebnisse landen
    'NEW_RUN_DIR': 'decision_tree_fresh_run'
}

# ----------------------------------------------------------------------------
# ---------------------- FUNKTIONEN ZUR DATENSAMMLUNG ------------------------
# ----------------------------------------------------------------------------
def collect_data(env, agent, num_episodes, max_steps):
    """
    Sammelt Daten mithilfe eines (bereits trainierten) PPO-Agenten.
    Gibt einen DataFrame mit Spalten ['state', 'action', 'reward'] zurück.
    """
    data = []
    for episode in range(num_episodes):
        reset_output = env.reset()
        if isinstance(reset_output, tuple):
            # neuere Gym-Version
            state, _ = reset_output
        else:
            # ältere Gym-Version
            state = reset_output
        
        for step in range(max_steps):
            # PPO-Agent wählt Aktion
            action, _ = agent.predict(state, deterministic=True)
            action = int(action)
            
            step_output = env.step(action)
            if len(step_output) == 5:
                next_state, reward, terminated, truncated, _ = step_output
                done = terminated or truncated
            else:
                next_state, reward, done, _ = step_output
            
            data.append([state, action, reward])
            state = next_state
            if done:
                break
    
    df = pd.DataFrame(data, columns=['state', 'action', 'reward'])
    return df

# ----------------------------------------------------------------------------
# ------------------- FUNKTIONEN FÜR DATENAUFBEREITUNG -----------------------
# ----------------------------------------------------------------------------
def preprocess_data(df):
    """
    Erwartet einen DataFrame mit Spalte 'state' (Zustandsvektor),
    erstellt separate Spalten für jeden Zustandswert,
    konvertiert 'action' zu int, entfernt NaNs.
    """
    # State in einzelne Features zerlegen
    state_features = pd.DataFrame(
        df['state'].tolist(), 
        columns=['x', 'y', 'vx', 'vy', 'theta', 'v_theta', 'left_leg', 'right_leg']
    )
    
    # Zusammenführen
    df = pd.concat([state_features, df[['action', 'reward']]], axis=1)
    
    # Aktion sicherheitshalber in int konvertieren
    df['action'] = df['action'].astype(int)
    
    # NaNs entfernen
    init_shape = df.shape
    df.dropna(inplace=True)
    final_shape = df.shape
    print(f"Datenbereinigung: Entfernte {init_shape[0] - final_shape[0]} Zeilen mit fehlenden Werten.")
    
    return df

# ----------------------------------------------------------------------------
# --------------- FUNKTIONEN FÜR TRAINING DER DECISION TREES ----------------
# ----------------------------------------------------------------------------
def train_decision_trees(X_train, y_train, depth, num_trees, random_state):
    """
    Trainiert mehrere Bäume derselben Tiefe und gibt sie als Liste zurück.
    """
    trees = []
    for i in range(num_trees):
        tree = DecisionTreeClassifier(
            max_depth=depth, 
            random_state=(random_state + i)
        )
        tree.fit(X_train, y_train)
        trees.append(tree)
    return trees

# ----------------------------------------------------------------------------
# ------------ FUNKTIONEN ZUR EVALUATION EINES DECISION TREES ---------------
# ----------------------------------------------------------------------------
def evaluate_tree_policy(tree, env, scaler, pca, selected_features, 
                         num_episodes, max_steps):
    """
    Bewertet die Performance (Mean Reward) eines Decision Trees als Policy 
    direkt in der angegebenen Gym-Umgebung.
    """
    rewards_all_episodes = []
    for _ in range(num_episodes):
        reset_output = env.reset()
        if isinstance(reset_output, tuple):
            state, _ = reset_output
        else:
            state = reset_output

        episode_reward = 0
        for _ in range(max_steps):
            # DataFrame mit dem aktuellen Zustand
            state_df = pd.DataFrame([state], columns=[
                'x', 'y', 'vx', 'vy', 'theta', 'v_theta', 'left_leg', 'right_leg'
            ])
            
            # Nur die "selected_features" nehmen
            state_selected = state_df[selected_features]

            # Skalieren + PCA
            state_scaled = scaler.transform(state_selected)
            state_pca = pca.transform(state_scaled)
            
            # Aktion vom Decision Tree
            action_pred = tree.predict(state_pca)[0]
            action = int(action_pred)
            action = np.clip(action, 0, env.action_space.n - 1)

            step_output = env.step(action)
            if len(step_output) == 5:
                next_state, reward, terminated, truncated, _ = step_output
                done = terminated or truncated
            else:
                next_state, reward, done, _ = step_output

            episode_reward += reward
            state = next_state
            
            if done:
                break
        
        rewards_all_episodes.append(episode_reward)
    
    mean_reward = np.mean(rewards_all_episodes)
    return mean_reward

# ----------------------------------------------------------------------------
# --------------------------------- MAIN -------------------------------------
# ----------------------------------------------------------------------------
def main():
    # ------------------ 1) Ordner-Struktur vorbereiten -----------------------
    NEW_RUN_DIR = CONFIG['NEW_RUN_DIR']
    os.makedirs(NEW_RUN_DIR, exist_ok=True)
    
    # ------------------ 2) Gym-Umgebung erstellen ---------------------------
    env = gym.make(CONFIG['ENV_NAME'])
    print(f"Erstelle Gym-Umgebung: {CONFIG['ENV_NAME']}")
    
    # ------------------ 3) PPO-Agent laden (optional) -----------------------
    model = None
    if os.path.exists(CONFIG['PPO_MODEL_PATH']):
        try:
            model = PPO.load(CONFIG['PPO_MODEL_PATH'])
            print(f"PPO-Modell erfolgreich geladen: {CONFIG['PPO_MODEL_PATH']}")
        except Exception as e:
            print(f"Fehler beim Laden des PPO-Modells: {e}")
    else:
        print("Kein PPO-Modellpfad gefunden. Bitte anpassen, falls erforderlich.")
    
    # ------------------ 4) Datensammlung (falls PPO vorhanden) --------------
    if model is not None:
        print("\nSammle Daten mit dem PPO-Agenten...")
        df = collect_data(
            env, 
            model, 
            num_episodes=CONFIG['NUM_EPISODES_COLLECT'], 
            max_steps=CONFIG['MAX_STEPS_COLLECT']
        )
    else:
        print("Kein PPO-Modell vorhanden, bitte Daten anderweitig bereitstellen.")
        return
    
    print(f"Größe des ursprünglichen DataFrames: {df.shape}")
    
    # ------------------ 5) Daten aufbereiten --------------------------------
    df = preprocess_data(df)
    print(f"Größe nach Preprocessing: {df.shape}")
    
    # ------------------ 6) Feature-Auswahl ----------------------------------
    # Wir haben 8 mögliche Zustands-Features (x, y, vx, vy, theta, v_theta, left_leg, right_leg)
    all_features = ['x', 'y', 'vx', 'vy', 'theta', 'v_theta', 'left_leg', 'right_leg']
    
    X_full = df[all_features]
    y = df['action']
    
    # ------------------ 7) Feature Selection via Random Forest --------------
    print("\nFühre Feature Selection via Random Forest durch...")
    rf_selector = RandomForestClassifier(n_estimators=100, random_state=CONFIG['RANDOM_STATE'])
    rf_selector.fit(X_full, y)
    
    importances = rf_selector.feature_importances_
    # Sortiere Features nach absteigender Importance
    feature_importance_pairs = sorted(
        zip(all_features, importances), 
        key=lambda x: x[1], 
        reverse=True
    )
    
    # Top N Features auswählen
    top_features = [f[0] for f in feature_importance_pairs[:CONFIG['TOP_N_FEATURES']]]
    print("Feature-Ranking (alle):")
    for feat, imp in feature_importance_pairs:
        print(f"  {feat}: {imp:.4f}")
    print(f"\nTop {CONFIG['TOP_N_FEATURES']} Features: {top_features}")
    
    # Nur diese Top-Features nehmen wir für den nächsten Schritt (Scaler + PCA + Decision Tree)
    X_selected = X_full[top_features]
    
    # ------------------ 8) Scaler und PCA NEU FITTEN ------------------------
    print("\nFitte neuen Scaler und neue PCA auf den Top-Features...")
    scaler = StandardScaler()
    scaler.fit(X_selected)
    X_scaled = scaler.transform(X_selected)
    
    pca = PCA(n_components=CONFIG['N_PCA_COMPONENTS'])
    pca.fit(X_scaled)
    X_pca = pca.transform(X_scaled)
    
    print("Skalierung und PCA abgeschlossen.")
    print(f"Erklärte Varianz (PCA): {pca.explained_variance_ratio_}")
    
    # Artefakte speichern
    scaler_path = os.path.join(NEW_RUN_DIR, "scaler_new.joblib")
    pca_path = os.path.join(NEW_RUN_DIR, "pca_new.joblib")
    joblib.dump(scaler, scaler_path)
    joblib.dump(pca, pca_path)
    print(f"Gespeichert: {scaler_path}, {pca_path}")
    
    # ------------------ 9) Train/Test-Split ---------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X_pca, y, 
        test_size=CONFIG['TEST_SIZE'], 
        random_state=CONFIG['RANDOM_STATE']
    )
    
    print(f"Trainingsdaten: {X_train.shape}, Testdaten: {X_test.shape}")
    
    # ------------------ 10) Bäume trainieren + bewerten ---------------------
    best_rewards_per_depth = []
    
    for depth in CONFIG['TREE_DEPTHS']:
        print(f"\n--- Decision Trees mit Tiefe = {depth} ---")
        trees = train_decision_trees(
            X_train, y_train, 
            depth=depth, 
            num_trees=CONFIG['NUM_TREES_PER_DEPTH'],
            random_state=CONFIG['RANDOM_STATE']
        )

        best_reward_for_depth = float("-inf")
        best_tree_idx = -1
        
        # Alle Bäume speichern + evaluieren
        for idx, tree in enumerate(trees):
            tree_filename = f"decision_tree_depth_{depth}_tree_{idx}.joblib"
            tree_path = os.path.join(NEW_RUN_DIR, tree_filename)
            joblib.dump(tree, tree_path)
            
            # Baumstruktur als Text (PCA-Feature-Namen)
            pca_feature_names = [f"PC{i+1}" for i in range(CONFIG['N_PCA_COMPONENTS'])]
            tree_as_text = export_text(tree, feature_names=pca_feature_names)
            
            txt_filename = f"tree_depth_{depth}_idx_{idx}.txt"
            with open(os.path.join(NEW_RUN_DIR, txt_filename), "w") as f:
                f.write(tree_as_text)
            
            # Evaluation
            mean_reward = evaluate_tree_policy(
                tree=tree, 
                env=env, 
                scaler=scaler, 
                pca=pca, 
                selected_features=top_features,  # <-- Hier: wir übergeben die Top-Features
                num_episodes=30, 
                max_steps=1000
            )
            print(f"  > Tree {idx}: Reward = {mean_reward:.2f}")
            
            # Bester Baum für diese Tiefe?
            if mean_reward > best_reward_for_depth:
                best_reward_for_depth = mean_reward
                best_tree_idx = idx
        
        print(f"  >> Bester Baum für Tiefe {depth} ist Tree {best_tree_idx} mit Reward = {best_reward_for_depth:.2f}")
        best_rewards_per_depth.append(best_reward_for_depth)
    
    # ------------------ 11) Ergebnisse speichern + Plot ---------------------
    results_df = pd.DataFrame({
        'Depth': CONFIG['TREE_DEPTHS'],
        'Best Reward': best_rewards_per_depth
    })
    results_csv_path = os.path.join(NEW_RUN_DIR, "results_depth.csv")
    results_df.to_csv(results_csv_path, index=False)
    print(f"\nErgebnisse gespeichert in {results_csv_path}")
    
    # Plot
    plt.figure()
    plt.plot(CONFIG['TREE_DEPTHS'], best_rewards_per_depth)
    plt.xlabel("Tiefe des Baumes")
    plt.ylabel("Bester Mean Reward")
    plt.title("Decision Tree Tiefe vs. (Bester) Mean Reward")
    plt.show()
    
    # Umgebung schließen
    env.close()

# ----------------------------------------------------------------------------
# ------------------------------ SCRIPT START ---------------------------------
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    main()
