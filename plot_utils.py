import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors  # Korrekte Importierung von colors
import pandas as pd
import gymnasium as gym
from stable_baselines3 import PPO  # Annahme: Modell ist mit Stable Baselines3 trainiert
from sklearn.preprocessing import MinMaxScaler  # Für die Normalisierung
from sklearn.decomposition import PCA  # Für die PCA-Analyse
from matplotlib.lines import Line2D  # Für benutzerdefinierte Legenden

# Pfade für Speicherung
PLOT_DIR = "plots_detail_view"
DATA_DIR = "plot_data"
MODEL_PATH = "models/ppo-LunarLander-v3/best_model.zip"  # Pfad zu deinem vortrainierten Modell
NUM_EPISODES = 100  # Anzahl der Episoden zum Sammeln von Daten (verdoppelt)

def load_model(model_path):
    """
    Lädt das vortrainierte Modell.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modell nicht gefunden unter: {model_path}")
    model = PPO.load(model_path)
    return model

def collect_data(model, env_id="LunarLander-v3", num_episodes=100):
    """
    Führt das Modell in der Umgebung aus und sammelt Zustands-Aktions-Paare.
    """
    env = gym.make(env_id)
    data = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        step = 0  # Optional: Schrittzähler hinzufügen
        while not done:
            action, _ = model.predict(state, deterministic=True)
            data.append((state, action))
            state, reward, done, truncated, info = env.step(action)
            step += 1
        print(f"Episoden {episode + 1}/{num_episodes} abgeschlossen, Schritte: {step}")  # Fortschritt anzeigen
    
    env.close()
    return data

def save_plot_data_and_generate(data, labels):
    """
    Erstellt für jedes Paar von Zustandsdimensionen einen separaten Plot,
    speichert ihn in ./plots_detail_view/ und exportiert die Daten als CSV.
    Außerdem wird eine PCA-Analyse durchgeführt und entsprechende Plots erstellt.
    Features werden normalisiert und Farben sind deutlich unterscheidbar.
    Zusätzlich werden die PCA-Loadings berechnet und visualisiert.
    Meta-Features (PC1 und PC2) werden als separate CSV gespeichert.
    """
    os.makedirs(PLOT_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    
    num_features = len(labels)
    
    # Organisiere die Daten in ein DataFrame
    states = np.array([item[0] for item in data])
    actions = np.array([item[1] for item in data])
    df = pd.DataFrame(states, columns=labels)
    df['action'] = actions
    
    # Normalisiere die Features
    scaler = MinMaxScaler()
    df_normalized = df.copy()
    df_normalized[labels] = scaler.fit_transform(df[labels])
    
    # Speichere das gesamte Datenset als CSV
    df_normalized.to_csv(os.path.join(DATA_DIR, "all_data_normalized.csv"), index=False)
    
    # Definiere eine Liste von klar unterscheidbaren Farben für die Aktionen
    action_colors = {
        0: 'blue',     # do nothing
        1: 'green',    # fire left engine
        2: 'orange',   # fire main engine
        3: 'red'        # fire right engine
    }
    # Erstelle eine benutzerdefinierte Colormap
    cmap = mcolors.ListedColormap([action_colors[key] for key in sorted(action_colors.keys())])
    
    # Für jedes Paar von Features, erstelle einen Plot
    for x_index in range(num_features):
        for y_index in range(x_index + 1, num_features):
            x_label = labels[x_index]
            y_label = labels[y_index]
            x = df_normalized[x_label]
            y = df_normalized[y_label]
            actions = df_normalized['action']
            
            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(x, y, c=actions, cmap=cmap, alpha=0.6, edgecolor='k')
            
            # Erstelle eine benutzerdefinierte Legende
            legend_elements = [
                Line2D([0], [0], marker='o', color='w', label='0: do nothing',
                       markerfacecolor=action_colors[0], markersize=10, markeredgecolor='k'),
                Line2D([0], [0], marker='o', color='w', label='1: fire left engine',
                       markerfacecolor=action_colors[1], markersize=10, markeredgecolor='k'),
                Line2D([0], [0], marker='o', color='w', label='2: fire main engine',
                       markerfacecolor=action_colors[2], markersize=10, markeredgecolor='k'),
                Line2D([0], [0], marker='o', color='w', label='3: fire right engine',
                       markerfacecolor=action_colors[3], markersize=10, markeredgecolor='k')
            ]
            plt.legend(handles=legend_elements, title="Aktionen", loc="upper right")
            
            plt.title(f"{x_label} vs {y_label} (normalisiert)")
            plt.xlabel(f"{x_label} (normalisiert)")
            plt.ylabel(f"{y_label} (normalisiert)")
            plt.grid(True)
            
            # Speichern des Plots
            plot_filename = f"{x_label}_vs_{y_label}.png"
            plt.savefig(os.path.join(PLOT_DIR, plot_filename))
            plt.close()
    
    # PCA-Analyse durchführen
    print("Führe PCA-Analyse durch...")
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(df_normalized[labels])
    pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
    pca_df['action'] = actions
    
    # Speichere die PCA-Daten
    pca_df.to_csv(os.path.join(DATA_DIR, "all_data_pca.csv"), index=False)
    
    # Plot der PCA-Ergebnisse
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(pca_df['PC1'], pca_df['PC2'], c=pca_df['action'], cmap=cmap, alpha=0.6, edgecolor='k')
    
    # Erstelle eine benutzerdefinierte Legende
    plt.legend(handles=legend_elements, title="Aktionen", loc="upper right")
    
    plt.title("PCA der Zustandsmerkmale (2 Hauptkomponenten)")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.2f}% Varianz)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.2f}% Varianz)")
    plt.grid(True)
    
    # Speichern des PCA-Plots
    pca_plot_filename = "PCA_plot.png"
    plt.savefig(os.path.join(PLOT_DIR, pca_plot_filename))
    plt.close()
    
    print("PCA-Analyse abgeschlossen und Plot gespeichert.")
    
    # Berechnung der Loadings
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    loading_df = pd.DataFrame(loadings, index=labels, columns=['PC1', 'PC2'])
    print("Loadings:")
    print(loading_df)
    
    # Speichere die Loadings als CSV
    loading_df.to_csv(os.path.join(DATA_DIR, "pca_loadings.csv"))
    
    # Plot der Loadings
    plt.figure(figsize=(10, 8))
    loading_df.plot(kind='bar', figsize=(10, 8))
    plt.title("PCA-Loadings der Zustandsmerkmale")
    plt.xlabel("Features")
    plt.ylabel("Loadings")
    plt.legend(title="Hauptkomponenten")
    plt.tight_layout()
    loading_plot_filename = "PCA_Loadings.png"
    plt.savefig(os.path.join(PLOT_DIR, loading_plot_filename))
    plt.close()
    
    print("Loadings berechnet, gespeichert und geplottet.")
    
    # Speichern der Meta-Features (PC1 und PC2) für spätere Verwendung in Decision Trees
    meta_features = pca_df[['PC1', 'PC2', 'action']]
    meta_features.to_csv(os.path.join(DATA_DIR, "meta_features.csv"), index=False)
    
    print("Meta-Features (PC1 und PC2) als 'meta_features.csv' gespeichert.")

def main():
    # Labels der Zustandsmerkmale entsprechend der LunarLander-Umgebung
    labels = [
        "x_position", "y_position", "x_velocity", "y_velocity",
        "angle", "angular_velocity", "left_leg_contact", "right_leg_contact"
    ]
    
    print("Lade das vortrainierte Modell...")
    model = load_model(MODEL_PATH)
    
    print(f"Sammle Daten aus {NUM_EPISODES} Episoden...")
    data = collect_data(model, num_episodes=NUM_EPISODES)
    print(f"Gesammelte {len(data)} Zustands-Aktions-Paare.")
    
    print("Generiere Plots und speichere Daten...")
    save_plot_data_and_generate(data, labels)
    print(f"Plots gespeichert in '{PLOT_DIR}' und Daten in '{DATA_DIR}'.")
    print("Fertig!")

if __name__ == "__main__":
    main()
