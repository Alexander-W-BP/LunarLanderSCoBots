import gym
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeClassifier, export_text
import warnings
import joblib
import os

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

        if (episode + 1) % 50 == 0:
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
    state_features = pd.DataFrame(
        df['state'].tolist(),
        columns=['x', 'y', 'vx', 'vy', 'theta', 'v_theta', 'left_leg', 'right_leg']
    )
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
    eines Random Forest Classifiers aus.

    Args:
        df: Der bereinigte DataFrame.
        target: Die Zielvariable ('action' oder 'reward').
        top_k: Anzahl der zu wählenden Top-Features.

    Returns:
        Tuple aus (X_selected, selected_features), wobei X_selected die
        Daten mit den wichtigsten Features enthält und selected_features
        die entsprechenden Spaltennamen sind.
    """
    X = df.drop(['action', 'reward'], axis=1)
    y = df[target]

    # Überprüfen der einzigartigen Werte in y
    unique_labels = y.unique()
    print(f"Einzigartige Labels in '{target}': {unique_labels}")

    # Falls y nicht Integer ist, konvertieren
    if not np.issubdtype(y.dtype, np.integer):
        print(f"Warnung: Die Zielvariable '{target}' ist nicht vom Typ Integer.")
        y = y.astype(int)
        print(f"Zielvariable '{target}' wurde in Integer konvertiert.")

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)

    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    feature_names = X.columns

    # Auswahl der Top-k Features
    selected_features = feature_names[indices[:top_k]]
    print(f"Ausgewählte Features: {list(selected_features)}")

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
        Tuple aus (X_pca, scaler, pca):
        - X_pca: Die PCA-transformierten Daten
        - scaler: Der Skalierer
        - pca: Das PCA-Modell
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=variance_threshold, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    print(f"Anzahl der PCA-Komponenten: {pca.n_components_}")
    return X_pca, scaler, pca


# ---------------------------
# Schritt 5: Training pro Tiefe
# ---------------------------

def train_decision_tree_for_depth(X_train, y_train, depth):
    """
    Führt ein GridSearchCV durch, um die besten Hyperparameter
    für einen DecisionTreeClassifier zu finden, wobei die Baumtiefe
    (max_depth) festgesetzt ist.

    Args:
        X_train: Trainingsdaten (Features).
        y_train: Trainingsdaten (Labels).
        depth: Feste max_depth des Entscheidungsbaums.

    Returns:
        Bestes DecisionTreeClassifier-Modell (für die gegebene Tiefe).
    """
    # Wir definieren ein Grid nur für min_samples_split und min_samples_leaf.
    # max_depth wird fest auf 'depth' gesetzt.
    param_grid = {
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 5]
    }

    # Basis-Modell mit fixer Tiefe
    tree = DecisionTreeClassifier(random_state=42, max_depth=depth)

    # GridSearchCV
    grid_search = GridSearchCV(
        tree,
        param_grid,
        cv=3,         # z.B. 3-fache Kreuzvalidierung
        n_jobs=-1,    # Alle verfügbaren CPU-Kerne
        verbose=1
    )
    grid_search.fit(X_train, y_train)

    best_tree = grid_search.best_estimator_
    print(f"[Tiefe {depth}] Beste Hyperparameter gefunden: {grid_search.best_params_}")
    return best_tree


# ---------------------------
# Schritt 6: Evaluation
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
            # Datenvorverarbeitung
            state_df = pd.DataFrame([state],
                                    columns=['x', 'y', 'vx', 'vy', 'theta', 'v_theta', 'left_leg', 'right_leg'])
            state_selected = state_df[selected_features]
            state_scaled = scaler.transform(state_selected)
            state_pca = pca.transform(state_scaled)

            # Aktion vorhersagen
            action_pred = tree.predict(state_pca)[0]
            action = int(action_pred)  # Aktion muss ganzzahlig sein
            action = np.clip(action, 0, env.action_space.n - 1)  # Gültige Aktion

            # Aktion ausführen
            step_output = env.step(action)
            if len(step_output) == 5:
                next_state, reward, terminated, truncated, _ = step_output
                done = terminated or truncated
            else:
                next_state, reward, done, _ = step_output

            total_reward += reward
            state = next_state

            if done:
                break

        total_rewards.append(total_reward)

        if (episode + 1) % 10 == 0:
            print(f"Evaluierung: Episode {episode+1}/{num_episodes} abgeschlossen.")

    mean_reward = np.mean(total_rewards)
    return mean_reward


# ---------------------------
# Schritt 7: Hauptfunktion
# ---------------------------

def main():
    # Initialisiere die LunarLander-Umgebung
    env = gym.make('LunarLander-v2')

    # Schritt 1: Laden des PPO-Modells
    MODEL_DIR = "models/ppo_LunarLander-v2/ppo-LunarLander-v2.zip"
    print("Schritt 1: Laden des PPO-Modells...")
    try:
        model = PPO.load(MODEL_DIR)
        print("PPO-Modell erfolgreich geladen.")
    except Exception as e:
        print(f"Fehler beim Laden des PPO-Modells: {e}")
        return

    # Schritt 2: Datensammlung
    print("\nSchritt 2: Datensammlung...")
    df = collect_data(env, model, num_episodes=100, max_steps=1000)
    print(f"Gesammelte Daten: {df.shape[0]} Zeilen und {df.shape[1]} Spalten.")

    # Schritt 3: Datenaufbereitung
    print("\nSchritt 3: Datenaufbereitung...")
    df = preprocess_data(df)
    print(f"Bereinigte Daten: {df.shape[0]} Zeilen und {df.shape[1]} Spalten.")

    # Schritt 4: Feature Selection
    print("\nSchritt 4: Feature Selection...")
    X_selected, selected_features = feature_selection(df, top_k=5, target='action')

    # Schritt 5: Dimensionsreduktion mit PCA
    print("\nSchritt 5: Dimensionsreduktion mit PCA...")
    X_pca, scaler, pca = apply_pca(X_selected, variance_threshold=0.95)

    # Vorbereitung der Daten
    X = X_pca
    y = df['action']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Trainingsdaten: {X_train.shape[0]} Zeilen, Testdaten: {X_test.shape[0]} Zeilen.")

    # Schritt 6: Decision Trees für Tiefe 3, 4 und 5 trainieren
    depths = [3, 4, 5]
    best_trees = {}
    for depth in depths:
        print(f"\nStarte Training für Tiefe = {depth}...")
        best_tree = train_decision_tree_for_depth(X_train, y_train, depth)
        best_trees[depth] = best_tree

        test_accuracy = best_tree.score(X_test, y_test)
        print(f"[Tiefe {depth}] Test-Genauigkeit (Accuracy): {test_accuracy:.4f}")

    # Schritt 7: Evaluation der Decision Trees im LunarLander
    print("\nEvaluierung der Decision Trees im LunarLander-Umfeld...")
    mean_rewards = {}
    for depth, tree in best_trees.items():
        print(f"\nEvaluierung des Decision Trees mit Tiefe {depth}...")
        mean_reward = evaluate_tree_policy(
            tree,
            scaler,
            pca,
            env,
            selected_features,
            num_episodes=30,   # Anzahl der Test-Episoden
            max_steps=1000
        )
        mean_rewards[depth] = mean_reward
        print(f"Mean Reward (Tiefe {depth}) über 30 Episoden: {mean_reward}")

    # Ermittlung des besten Mean Rewards
    best_depth = max(mean_rewards, key=mean_rewards.get)
    best_mean_reward = mean_rewards[best_depth]
    print("\n-------------------------------------------")
    print(f"Beste Tiefe: {best_depth} mit Mean Reward = {best_mean_reward}")
    print("-------------------------------------------")

    # Modelle und Artefakte speichern
    MODEL_SAVE_DIR = "decision_trees"
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

    # Speichere jeden Baum als Text und Joblib
    pca_feature_names = [f'PC{i}' for i in range(1, pca.n_components_ + 1)]
    for depth, tree in best_trees.items():
        # Export als Text
        tree_text = export_text(tree, feature_names=pca_feature_names)
        with open(os.path.join(MODEL_SAVE_DIR, f'decision_tree_depth_{depth}.txt'), 'w') as f:
            f.write(tree_text)
        print(f"Entscheidungsbaum (Tiefe {depth}) als Textdatei gespeichert (decision_tree_depth_{depth}.txt).")

        # Export als Joblib
        joblib.dump(tree, os.path.join(MODEL_SAVE_DIR, f'decision_tree_depth_{depth}.joblib'))
        print(f"Entscheidungsbaum (Tiefe {depth}) als Joblib-Datei gespeichert (decision_tree_depth_{depth}.joblib).")

    # Preprocessing-Artefakte speichern (gemeinsam für alle Bäume)
    joblib.dump(scaler, os.path.join(MODEL_SAVE_DIR, 'scaler.joblib'))
    joblib.dump(pca, os.path.join(MODEL_SAVE_DIR, 'pca.joblib'))
    joblib.dump(selected_features, os.path.join(MODEL_SAVE_DIR, 'selected_features.joblib'))
    print("Preprocessing-Artefakte (Scaler, PCA, Features) gespeichert.")

    # Speichern der Ergebnisse in CSV
    results_data = []
    for d in depths:
        acc = best_trees[d].score(X_test, y_test)
        results_data.append({
            'Depth': d,
            'Mean Reward': mean_rewards[d],
            'Test Accuracy': acc
        })
    results_df = pd.DataFrame(results_data)
    results_df.to_csv('decision_tree_evaluation_results.csv', index=False)
    print("\nEvaluationsergebnisse in 'decision_tree_evaluation_results.csv' gespeichert.")

    env.close()


if __name__ == "__main__":
    main()
