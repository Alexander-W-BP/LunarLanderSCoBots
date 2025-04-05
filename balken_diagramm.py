import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_rewards_comparison(files_to_compare):
    """
    Plottet ein Balkendiagramm (mit Fehlerbalken) zum Vergleich verschiedener CSV-Dateien.
    
    Parameters:
    -----------
    files_to_compare : list of str
        Liste mit den Schlüsseln, die im 'all_files'-Dictionary vorhanden sind.
        Beispiel: ['original_features', 'llm_features'] oder ['only_pca'] usw.
    """
    
    # Verfügbarer Pool an Dateien mit passendem Label und Farbe:
    # Du kannst die Namen, Pfade und Farben hier anpassen.
    all_files = {
        "original_features": {
            "path": "rewards/original_features.csv",
            "label": "Original Features",
            "color": "#1f77b4"
        },
        "llm_features": {
            "path": "rewards/chatgpt.csv",
            "label": "LLM Features",
            "color": "#ff7f0e"
        },
        "only_pca": {
            "path": "rewards/only_pca.csv",
            "label": "Only PCA-Features",
            "color": "#2ca02c"
        },
        "meta_features": {
            "path": "rewards/all_features.csv",
            "label": "PCA-derived Meta-Features",
            "color": "#d62728"
        },
        "plot_features": {
            "path": "rewards/plot_features.csv",
            "label": "Action Space Division Features",
            "color": "#9467bd"
        },
        "top5": {
            "path": "rewards/top_5.csv",
            "label": "Top-5",
            "color": "#8c564b"
        },
    }
    
    # Ausgewählte Einträge filtern
    chosen = [all_files[key] for key in files_to_compare if key in all_files]
    
    # Falls versehentlich ein ungültiger Schlüssel übergeben wurde, könnte chosen leer sein.
    if not chosen:
        print("Keine gültigen Dateien ausgewählt. Bitte überprüfe 'files_to_compare'.")
        return
    
    # Die erste gewählte Datei lesen wir ein, um die 'depths' zu bestimmen
    # (wir gehen davon aus, dass in allen CSV-Dateien die gleichen depths stehen).
    df_first = pd.read_csv(chosen[0]['path'])
    depths = df_first['depths'].values
    x = np.arange(len(depths))  # Positionen auf der x-Achse
    
    # Figure erstellen
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Vorbereitung für mehrere Balken
    n = len(chosen)         # Anzahl der ausgewählten Dateien
    bar_width = 0.8 / n     # Gemeinsame Balkenbreite, damit alle Platz finden
    total_width = 0.8
    bar_width = total_width / n
    offset = (np.arange(n) - (n - 1) / 2) * bar_width
    
    # Balken für jede gewählte CSV-Datei (mit Fehlerbalken)
    for i, info in enumerate(chosen):
        # CSV einlesen
        df = pd.read_csv(info['path'])
        
        # Mean + STD aus DataFrame
        means = df['mean_rewards'].values
        stds  = df['std_rewards'].values
        
        # Versatz für die Gruppierung festlegen
        # offset[i] verschiebt jeden Balken nach links oder rechts
        ax.bar(x + offset[i], means,
               width=bar_width,
               yerr=stds,     # Fehlermarge
               capsize=4,     # "Käppchen" an den Fehlerbalken
               label=info['label'],
               color=info['color'])
    
    # Achsenbeschriftungen
    ax.set_xticks(x)
    ax.set_xticklabels(depths)
    ax.set_xlabel('Tiefe (depth)')
    ax.set_ylabel('Mean Reward')
    
    # Titel (optional anpassbar)
    ax.set_title('Vergleich der Methoden nach Tiefe')
    ax.axhline(y=100, color='gray', linestyle='--', linewidth=1)
    ax.axhline(y=200, color='gray', linestyle='--', linewidth=1)
    
    # Legende
    ax.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # Hier stellst du flexibel ein, was du vergleichen willst:
    # Beispiele:
    #files_to_compare = ["original_features"]  # Nur Original Features
    files_to_compare = ["llm_features",  "meta_features", "plot_features"]  # Only PCA und Meta-Features
    #files_to_compare = ["original_features", "llm_features", "only_pca"]  # Drei Methoden
    #files_to_compare = ["original_features", "llm_features", "only_pca", "meta_features"]  # Alle vier
    
    plot_rewards_comparison(files_to_compare)
