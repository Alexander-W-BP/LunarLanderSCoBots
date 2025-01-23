import gym
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier, export_text
import joblib
import os
import warnings
import time
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# ---------------------------
# Evaluation und Visualisierung
# ---------------------------

def preprocess_state(state, scaler, pca, selected_features):
    """
    Wendet die gleichen Vorverarbeitungsschritte an wie beim Training.

    Args:
        state: Der aktuelle Zustand der Umgebung.
        scaler: Der geladene StandardScaler.
        pca: Das geladene PCA-Modell.
        selected_features: Die geladenen ausgewählten Features.

    Returns:
        Transformierter Zustand.
    """
    # Umwandlung des Zustands in einen DataFrame
    state_df = pd.DataFrame([state], columns=['x', 'y', 'vx', 'vy', 'theta', 'v_theta', 'left_leg', 'right_leg'])
    
    # Auswahl der relevanten Features
    state_selected = state_df[selected_features]
    
    # Skalierung der Features
    state_scaled = scaler.transform(state_selected)
    
    # Anwendung von PCA
    state_pca = pca.transform(state_scaled)
    
    return state_pca

def print_pca_components(pca, selected_features):
    """
    Gibt die linearen Kombinationen der PCA-Komponenten aus.

    Args:
        pca: Das geladene PCA-Modell.
        selected_features: Die Namen der ursprünglichen Features.
    """
    print("----- PCA Komponenten (Linearkombinationen der ursprünglichen Features) -----")
    for idx, component in enumerate(pca.components_):
        component_str = " + ".join([f"{weight:.4f}*{feature}" for weight, feature in zip(component, selected_features)])
        print(f"PC{idx + 1}: {component_str}")
    print("---------------------------------------------------------------------------\n")

def print_decision_tree_text(tree, feature_names):
    """
    Gibt den Entscheidungsbaum als Text aus.

    Args:
        tree: Das geladene DecisionTreeClassifier-Modell.
        feature_names: Die Namen der Features, die im Baum verwendet werden.
    """
    tree_rules = export_text(tree, feature_names=feature_names)
    print("----- Entscheidungsbaum (auf PCA-transformierten Features) -----")
    print(tree_rules)
    print("------------------------------------------------------------\n")

def analyze_pca_components(pca, selected_features):
    """
    Analysiert die PCA-Komponenten und identifiziert die wichtigsten ursprünglichen Features.

    Args:
        pca: Das geladene PCA-Modell.
        selected_features: Die Namen der ursprünglichen Features.
    """
    print("----- Analyse der PCA-Komponenten -----")
    for idx, component in enumerate(pca.components_):
        feature_contributions = pd.Series(component, index=selected_features)
        top_features = feature_contributions.abs().sort_values(ascending=False).head(3)
        print(f"PC{idx + 1} - Top 3 Features:")
        for feature, weight in top_features.items():
            print(f"   {feature}: {weight:.4f}")
        print()
    print("---------------------------------------\n")

def plot_explained_variance(pca):
    """
    Plottet die erklärte Varianz jeder PCA-Komponente.

    Args:
        pca: Das geladene PCA-Modell.
    """
    plt.figure(figsize=(10, 6))
    components = np.arange(1, len(pca.explained_variance_ratio_) + 1)
    sns.barplot(x=components, y=pca.explained_variance_ratio_ * 100, color='skyblue')
    plt.plot(components, np.cumsum(pca.explained_variance_ratio_) * 100, marker='o', color='red', label='Kumulative erklärte Varianz')
    for i, v in enumerate(pca.explained_variance_ratio_ * 100):
        plt.text(i + 1, v + 0.5, f"{v:.1f}%", ha='center', va='bottom', fontsize=9)
    plt.xlabel('PCA-Komponente')
    plt.ylabel('Erklärte Varianz (%)')
    plt.title('Erklärte Varianz jeder PCA-Komponente')
    plt.xticks(components)
    plt.ylim(0, max(pca.explained_variance_ratio_ * 100) + 10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_pca_loadings(pca, selected_features):
    """
    Plottet die Loadings jeder PCA-Komponente.

    Args:
        pca: Das geladene PCA-Modell.
        selected_features: Die Namen der ursprünglichen Features.
    """
    num_components = pca.n_components_
    num_features = len(selected_features)

    # Erstellen eines DataFrames für die Loadings
    loadings = pd.DataFrame(pca.components_.T, columns=[f'PC{i+1}' for i in range(num_components)], index=selected_features)

    # Setze das Layout für die Subplots
    fig, axes = plt.subplots(nrows=num_components, ncols=1, figsize=(12, 5 * num_components))
    
    if num_components == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        sns.barplot(x=loadings.index, y=loadings.iloc[:, i], palette='viridis', ax=ax)
        ax.set_title(f'Loadings für PC{i+1}')
        ax.set_xlabel('Original Features')
        ax.set_ylabel('Loading')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Hinzufügen von Annotations
        for p in ax.patches:
            height = p.get_height()
            ax.annotate(f'{height:.2f}', 
                        (p.get_x() + p.get_width() / 2., height), 
                        ha='center', va='bottom', 
                        fontsize=10, color='black', 
                        xytext=(0, 5), 
                        textcoords='offset points')

    plt.tight_layout()
    plt.show()

def plot_decision_tree_importances(tree, feature_names):
    """
    Plottet die Feature-Importances des Entscheidungsbaums.

    Args:
        tree: Das geladene DecisionTreeClassifier-Modell.
        feature_names: Die Namen der Features, die im Baum verwendet werden.
    """
    importances = tree.feature_importances_
    feature_importances = pd.Series(importances, index=feature_names).sort_values(ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=feature_importances.values, y=feature_importances.index, palette='magma')
    plt.xlabel('Feature Importance')
    plt.ylabel('PCA-Komponente')
    plt.title('Feature-Importances des Entscheidungsbaums')
    plt.xlim(0, 1)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Hinzufügen von Annotations
    for i, v in enumerate(feature_importances.values):
        plt.text(v + 0.01, i, f"{v:.2f}", va='center', fontsize=9)
    
    plt.tight_layout()
    plt.show()

def main():
    # Pfad zum Verzeichnis, in dem die Modelle und Preprocessing-Artefakte gespeichert sind
    MODEL_DIR = "decision_trees_best"  # Passe diesen Pfad bei Bedarf an

    # Überprüfen, ob das Modellverzeichnis existiert
    if not os.path.exists(MODEL_DIR):
        print(f"Modellverzeichnis '{MODEL_DIR}' existiert nicht. Stelle sicher, dass die Modelle vorhanden sind.")
        return

    # Laden der Preprocessing-Artefakte
    try:
        scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.joblib'))
        pca = joblib.load(os.path.join(MODEL_DIR, 'pca.joblib'))
        selected_features = joblib.load(os.path.join(MODEL_DIR, 'selected_features.joblib'))
        print("Preprocessing-Artefakte erfolgreich geladen.\n")
    except Exception as e:
        print(f"Fehler beim Laden der Preprocessing-Artefakte: {e}")
        return

    # Laden des Decision Trees mit Tiefe 3
    tree_path = os.path.join(MODEL_DIR, 'decision_tree_depth_3_168.63.joblib')
    if not os.path.exists(tree_path):
        print(f"Decision Tree mit Tiefe 3 nicht gefunden unter '{tree_path}'. Stelle sicher, dass die Datei existiert.")
        return

    try:
        tree = joblib.load(tree_path)
        print("Decision Tree mit Tiefe 3 erfolgreich geladen.\n")
    except Exception as e:
        print(f"Fehler beim Laden des Decision Trees: {e}")
        return

    # Ausgabe der PCA-Komponenten
    print_pca_components(pca, selected_features)

    # Analyse der PCA-Komponenten
    analyze_pca_components(pca, selected_features)

    # Visualisierung der erklärten Varianz
    plot_explained_variance(pca)

    # Visualisierung der PCA Loadings
    plot_pca_loadings(pca, selected_features)

    # Visualisierung der Feature-Importances des Entscheidungsbaums
    plot_decision_tree_importances(tree, [f"PC{i+1}" for i in range(pca.n_components_)])

    # Ausgabe des Entscheidungsbaums
    print_decision_tree_text(tree, [f"PC{i+1}" for i in range(pca.n_components_)])

    # Initialisieren der LunarLander-Umgebung mit Rendering
    env = gym.make('LunarLander-v2', render_mode='human')

    num_episodes = 3  # Anzahl der durchzuführenden Episoden
    max_steps = 1000   # Maximale Schritte pro Episode

    for episode in range(1, num_episodes + 1):
        print(f"\nEpisode {episode}/{num_episodes} startet...")
        
        reset_output = env.reset()
        if isinstance(reset_output, tuple):
            state, _ = reset_output  # Entpacken des Tupels (state, info)
        else:
            state = reset_output

        total_reward = 0
        done = False
        step = 0

        while not done and step < max_steps:
            # Preprocessing des Zustands
            state_pca = preprocess_state(state, scaler, pca, selected_features)
            
            # Vorhersage der Aktion
            action_pred = tree.predict(state_pca)[0]
            action = int(action_pred)  # Aktionen sind diskret (0, 1, 2, 3)
            action = np.clip(action, 0, env.action_space.n - 1)  # Sicherstellen, dass Aktion gültig ist

            # Aktion ausführen
            step_output = env.step(action)
            if len(step_output) == 5:
                next_state, reward, terminated, truncated, _ = step_output
                done = terminated or truncated
            else:
                next_state, reward, done, _ = step_output  # Fallback für ältere Gym-Versionen

            total_reward += reward
            state = next_state
            step += 1

            # Kurze Pause, um die Rendering-Geschwindigkeit zu steuern
            time.sleep(0.02)  # 20 ms Pause

        print(f"Episode {episode} abgeschlossen. Gesamte Belohnung: {total_reward}")

    env.close()
    print("\nAlle Episoden abgeschlossen.")

if __name__ == "__main__":
    main()
