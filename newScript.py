import gym
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text  # Standard Decision Tree
import warnings
import joblib
import os
import seaborn as sns

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Import PPO from stable_baselines3
from stable_baselines3 import PPO

# ---------------------------
# Schritt 1: Datensammlung
# ---------------------------

def collect_data(env, agent, num_episodes, max_steps):
    """
    Sammelt Daten von der Interaktion des PPO-Agenten mit der Umgebung.

    Args:
        env: Die Gym-Umgebung.
        agent: Der PPO-Agent.
        num_episodes: Anzahl der Episoden zur Datensammlung.
        max_steps: Maximale Schritte pro Episode.

    Returns:
        Pandas DataFrame mit den gesammelten Daten.
    """
    data = []
    for episode in range(num_episodes):
        reset_output = env.reset()
        if isinstance(reset_output, tuple):
            state, _ = reset_output  # Entpacken des Tupels (state, info)
        else:
            state = reset_output
        for step in range(max_steps):
            action, _ = agent.predict(state, deterministic=True)  # PPO-Modell zur Aktionsvorhersage
            action = int(action)  # Sicherstellen, dass Aktion ein Integer ist
            step_output = env.step(action)
            if len(step_output) == 5:
                next_state, reward, terminated, truncated, _ = step_output
                done = terminated or truncated
            else:
                next_state, reward, done, _ = step_output  # Fallback für ältere Gym-Versionen
            
            data.append([state, action, reward])
            state = next_state
            if done:
                break
        
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode+1}/{num_episodes} abgeschlossen.")
    
    df = pd.DataFrame(data, columns=['state', 'action', 'reward'])
    return df

# ---------------------------
# Schritt 2: Datenaufbereitung
# ---------------------------

def preprocess_data(df):
    """
    Zerlegt die Zustandsvektoren in einzelne Features und bereinigt die Daten.

    Args:
        df: Der ursprüngliche DataFrame mit gesammelten Daten.

    Returns:
        Bereinigter DataFrame mit getrennten Features.
    """
    # Zerlege den State-Vektor in einzelne Features
    state_features = pd.DataFrame(df['state'].tolist(), 
                                  columns=['x', 'y', 'v_x', 'v_y', 'angle', 'v_angle', 'left_leg', 'right_leg'])
    df = pd.concat([state_features, df[['action', 'reward']]], axis=1)
    
    # Sicherstellen, dass 'action' als Integer formatiert ist
    df['action'] = df['action'].astype(int)
    
    # Datenbereinigung
    initial_shape = df.shape
    df.dropna(inplace=True)
    final_shape = df.shape
    print(f"Datenbereinigung: Entfernte {initial_shape[0] - final_shape[0]} Zeilen mit fehlenden Werten.")
    
    return df

# ---------------------------
# Schritt 3: Feature Selection
# ---------------------------

def feature_selection(df, top_k, target='action'):
    """
    Wählt die wichtigsten Features basierend auf der Feature-Wichtigkeit 
    eines Random Forest Classifiers aus und visualisiert sie als Balkendiagramm
    mithilfe von Seaborn.

    Args:
        df: Der bereinigte DataFrame.
        target: Die Zielvariable ('action' oder 'reward').
        top_k: Anzahl der zu wählenden Top-Features.

    Returns:
        Tuple (X_selected, selected_features), wobei X_selected die ausgewählten 
        Feature-Daten enthält und selected_features die Namen dieser Features.
    """
    # X und y definieren
    X = df.drop(['action', 'reward'], axis=1)
    y = df[target]

    # Sicherstellen, dass y (die Zielvariable) Integer-Werte hat
    unique_labels = y.unique()
    print(f"Unique labels in '{target}': {unique_labels}")
    if not np.issubdtype(y.dtype, np.integer):
        print(f"Warnung: Die Zielvariable '{target}' ist nicht vom Typ Integer.")
        y = y.astype(int)
        print(f"Zielvariable '{target}' wurde in Integer konvertiert.")

    # Random Forest trainieren
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)

    # Feature Importances extrahieren und sortieren
    importances = rf.feature_importances_
    feature_names = X.columns
    feature_importances = pd.Series(importances, index=feature_names).sort_values(ascending=False)


    # Visualisierung mit Seaborn (viridis-Palette)
    plt.figure(figsize=(8, 5))
    sns.barplot(x=feature_importances.values, 
                y=feature_importances.index, 
                palette='viridis')
    plt.title("Feature Importance mittels Random Forest")
    plt.xlabel("Wichtigkeit")
    plt.ylabel("Features")
    plt.tight_layout()
    plt.show()

    # Nur die Top-K-Features auswählen
    selected_features = feature_importances.index[:top_k]
    print(f"\nSelected Features (Top {top_k}): {list(selected_features)}")

    # X_selected auf die Top-K-Features beschränken
    X_selected = X[selected_features]

    return X_selected, selected_features

# ---------------------------
# Schritt 4: Dimensionsreduktion mit PCA
# ---------------------------

def apply_pca(X, variance_threshold=0.95):
    """
    Wendet PCA zur Dimensionsreduktion an und behält einen bestimmten Varianzanteil.

    Args:
        X: Die ausgewählten Features.
        variance_threshold: Anteil der beibehaltenen Varianz.

    Returns:
        Tuple (X_pca, scaler, pca), wobei X_pca die transformierten Daten sind, 
        scaler der StandardScaler und pca das PCA-Objekt.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA(n_components=variance_threshold, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    
    print(f"Anzahl der PCA-Komponenten: {pca.n_components_}")
    return X_pca, scaler, pca

# ---------------------------
# Schritt 5: Training der Decision Trees
# ---------------------------

def train_decision_trees(X_train, y_train, depth, num_trees):
    """
    Trainiert mehrere Decision Trees mit derselben Tiefe.

    Args:
        X_train: Trainingsdaten.
        y_train: Trainingslabels.
        depth: Tiefe für den Entscheidungsbaum.
        num_trees: Anzahl der zu trainierenden Bäume.

    Returns:
        Liste der trainierten Bäume.
    """
    trees = []
    for i in range(num_trees):
        tree = DecisionTreeClassifier(max_depth=depth, random_state=i)
        tree.fit(X_train, y_train)
        trees.append(tree)
    return trees

# ---------------------------
# Schritt 6: Evaluation der Decision Trees
# ---------------------------

def evaluate_tree_policy(tree, scaler, pca, env, selected_features, num_episodes, max_steps):
    """
    Bewertet die Performance eines Decision Trees als Policy in der Umgebung.

    Args:
        tree: Der trainierte Decision Tree.
        scaler: Der Skalierer, der auf die Features angewendet wurde.
        pca: Das PCA-Modell, das auf die Features angewendet wurde.
        env: Die Gym-Umgebung.
        selected_features: Die ausgewählten Features.
        num_episodes: Anzahl der Episoden zur Evaluation.
        max_steps: Maximale Schritte pro Episode.

    Returns:
        Mean Reward über alle Episoden.
    """
    total_rewards = []
    
    for episode in range(num_episodes):
        reset_output = env.reset()
        if isinstance(reset_output, tuple):
            state, _ = reset_output  # Entpacken des Tupels (state, info)
        else:
            state = reset_output

        total_reward = 0
        for step in range(max_steps):
            # Datenvorverarbeitung für den aktuellen Zustand
            state_df = pd.DataFrame([state], 
                                    columns=['x', 'y', 'v_x', 'v_y', 
                                             'angle', 'v_angle', 
                                             'left_leg', 'right_leg'])
            state_selected = state_df[selected_features]
            state_scaled = scaler.transform(state_selected)
            state_pca = pca.transform(state_scaled)
            
            # Aktion vorhersagen
            action_pred = tree.predict(state_pca)[0]
            action = int(action_pred)  # Aktionen sind diskret (0, 1, 2, 3)
            action = np.clip(action, 0, env.action_space.n - 1)  # Gültigkeit sicherstellen
            
            # Aktion ausführen
            step_output = env.step(action)
            if len(step_output) == 5:
                next_state, reward, terminated, truncated, _ = step_output
                done = terminated or truncated
            else:
                next_state, reward, done, _ = step_output  # Fallback für ältere Gym-Versionen
            
            total_reward += reward
            state = next_state
            if done:
                break
        
        total_rewards.append(total_reward)
    
    mean_reward = np.mean(total_rewards)
    return mean_reward

# ---------------------------
# Hauptfunktion
# ---------------------------

def main():
    # Initialisiere die LunarLander-Umgebung
    env = gym.make('LunarLander-v2')
    
    # Schritt 1: Laden des PPO-Modells
    MODEL_DIR = "models\\ppo_LunarLander-v2\\ppo-LunarLander-v2.zip"
    print("Schritt 1: Laden des PPO-Modells...")
    try:
        model = PPO.load(MODEL_DIR)
        print("PPO-Modell erfolgreich geladen.")
    except Exception as e:
        print(f"Fehler beim Laden des PPO-Modells: {e}")
        return
    
    # Schritt 2: Datensammlung
    print("\nSchritt 2: Datensammlung...")
    df = collect_data(env, model, num_episodes=500, max_steps=1000)
    print(f"Gesammelte Daten: {df.shape[0]} Zeilen und {df.shape[1]} Spalten.")
    
    # Schritt 3: Datenaufbereitung
    print("\nSchritt 3: Datenaufbereitung...")
    df = preprocess_data(df)
    print(f"Bereinigte Daten: {df.shape[0]} Zeilen und {df.shape[1]} Spalten.")
    
    # Schritt 4: Feature Selection (Random Forest)
    print("\nStep 4: Feature Selection...")
    X_selected, selected_features = feature_selection(df, top_k=5, target='action')
    
    # Schritt 5: Dimensionsreduktion mit PCA
    print("\nSchritt 5: Dimensionsreduktion mit PCA...")
    X_pca, scaler, pca = apply_pca(X_selected, variance_threshold=0.95)
    
    # Erstelle Ordner für gespeicherte Modelle
    MODEL_SAVE_DIR = "decision_trees"
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    
    # Speichere Scaler, PCA und ausgewählte Features direkt nach der PCA
    joblib.dump(scaler, os.path.join(MODEL_SAVE_DIR, 'scaler.joblib'))
    joblib.dump(pca, os.path.join(MODEL_SAVE_DIR, 'pca.joblib'))
    joblib.dump(selected_features, os.path.join(MODEL_SAVE_DIR, 'selected_features.joblib'))
    print("Preprocessing-Artefakte (Scaler, PCA, ausgewählte Features) wurden gespeichert.")

    # Schritt 6: Training und Evaluation der Decision Trees (Tiefe 1 bis 4)
    print("\nSchritt 6: Training und Evaluation der Decision Trees...")
    X = X_pca
    y = df['action']
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size=0.2, 
                                                        random_state=42)
    
    print(f"Trainingsdaten: {X_train.shape[0]} Zeilen, Testdaten: {X_test.shape[0]} Zeilen.")
    
    # Liste, um den durchschnittlichen Reward pro Tiefe zu speichern
    mean_rewards_per_depth = []
    
    for depth in range(1, 5):
        print(f"\n--- Training von Entscheidungsbäumen mit Tiefe = {depth} ---")
        # Trainiere 10 Bäume pro Tiefe
        trees = train_decision_trees(X_train, y_train, depth=depth, num_trees=1)
        
        # Evaluierung und Speichern
        rewards_for_this_depth = []
        for idx, tree in enumerate(trees):
            mean_reward = evaluate_tree_policy(tree, scaler, pca, env, 
                                               selected_features, 
                                               num_episodes=30, 
                                               max_steps=1000)
            rewards_for_this_depth.append(mean_reward)
            print(f"  > Decision Tree {idx}: Mean Reward = {mean_reward}")
            
            # --- Speichern des Entscheidungsbaums als Textdatei ---
            pca_feature_names = [f'PC{i}' for i in range(1, pca.n_components_ + 1)]
            tree_text = export_text(tree, feature_names=pca_feature_names)
            txt_filename = f'decision_tree_depth_{depth}_tree_{idx}.txt'
            with open(txt_filename, 'w') as f:
                f.write(tree_text)
            
            # --- Speichern des Entscheidungsbaums als Joblib-Datei ---
            joblib_filename = os.path.join(MODEL_SAVE_DIR, 
                                           f'decision_tree_depth_{depth}_tree_{idx}.joblib')
            joblib.dump(tree, joblib_filename)
        
        # Durchschnittlichen Reward für diese Tiefe ermitteln
        avg_reward_this_depth = np.mean(rewards_for_this_depth)
        mean_rewards_per_depth.append(avg_reward_this_depth)
        print(f"Durchschnittlicher Reward (Tiefe {depth}): {avg_reward_this_depth}")
    
    env.close()
    
    # Schritt 7: Ergebnisse speichern
    results_df = pd.DataFrame({
        'Depth': range(1, 5),
        'Mean Reward': mean_rewards_per_depth
    })
    results_df.to_csv('decision_tree_evaluation_results.csv', index=False)
    print("\nErgebnisse wurden in 'decision_tree_evaluation_results.csv' gespeichert.")
    
    # Schritt 8: Plot der Rewards für jede Tiefe
    plt.figure()
    plt.plot(range(1, 5), mean_rewards_per_depth)
    plt.xlabel("Tiefe des Baumes")
    plt.ylabel("Mean Reward")
    plt.title("Entscheidungsbaum-Tiefe vs. Mean Reward")
    plt.show()

if __name__ == "__main__":
    main()
