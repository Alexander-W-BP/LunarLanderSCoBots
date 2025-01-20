import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib
import os

# 1. Daten Laden
def load_data(filepath):
    """
    Lädt die gesammelten Daten aus einer Pickle-Datei.

    Args:
        filepath (str): Pfad zur Pickle-Datei.

    Returns:
        pd.DataFrame: Geladener Datensatz.
    """
    try:
        df = pd.read_pickle(filepath)
        print(f"Daten erfolgreich geladen. Shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Fehler beim Laden der Daten: {e}")
        return None

# 2. PCA Analyse
def perform_pca(df, n_components=5, output_dir='pca_outputs'):
    """
    Führt PCA durch, extrahiert die Loadings und speichert die Ergebnisse.

    Args:
        df (pd.DataFrame): Datensatz mit Zuständen und Aktionen.
        n_components (int): Anzahl der Hauptkomponenten.
        output_dir (str): Verzeichnis zum Speichern der PCA-Ergebnisse.

    Returns:
        pd.DataFrame: DataFrame mit den PCA-Meta-Features.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("\n--- Principal Component Analysis (PCA) ---\n")
    
    # Vorbereitung der Daten (ohne Zielvariable)
    X = df.drop('action', axis=1)
    y = df['action']
    
    # Skalieren der Features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    scaler_filename = os.path.join(output_dir, 'scaler.pkl')
    joblib.dump(scaler, scaler_filename)
    print(f"Scaler gespeichert unter '{scaler_filename}'")
    
    # PCA durchführen
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    
    # Erklärte Varianz speichern
    explained_variance = pca.explained_variance_ratio_
    explained_variance_df = pd.DataFrame({
        'Principal Component': [f'PC{i}' for i in range(1, n_components+1)],
        'Explained Variance Ratio': explained_variance
    })
    explained_variance_df.to_csv(os.path.join(output_dir, 'explained_variance.csv'), index=False)
    print("Explained Variance Ratio gespeichert unter 'explained_variance.csv'")
    
    # Visualisierung der erklärten Varianz
    plt.figure(figsize=(8, 6))
    sns.barplot(x=explained_variance_df['Principal Component'], y=explained_variance_df['Explained Variance Ratio'], palette='viridis')
    plt.title("Erklärte Varianz der Hauptkomponenten")
    plt.xlabel("Hauptkomponente")
    plt.ylabel("Explained Variance Ratio")
    plt.savefig(os.path.join(output_dir, 'explained_variance.png'))
    plt.close()
    print("Plot 'explained_variance.png' gespeichert.")
    
    # Loadings extrahieren und speichern
    loadings = pd.DataFrame(pca.components_.T, columns=[f'PC{i}' for i in range(1, n_components+1)], index=X.columns)
    loadings.to_csv(os.path.join(output_dir, 'pca_loadings.csv'))
    print("PCA Loadings gespeichert unter 'pca_loadings.csv'")
    
    # Visualisierung der Loadings für jede Hauptkomponente
    for i in range(n_components):
        plt.figure(figsize=(10, 6))
        sns.barplot(x=loadings.iloc[:, i], y=loadings.index, palette='viridis')
        plt.title(f"PCA Loadings für PC{i+1}")
        plt.xlabel("Loading")
        plt.ylabel("Feature")
        plt.savefig(os.path.join(output_dir, f'pca_loadings_PC{i+1}.png'))
        plt.close()
        print(f"Plot 'pca_loadings_PC{i+1}.png' gespeichert.")
    
    # DataFrame der PCA-Meta-Features erstellen
    pca_columns = [f'PC{i}' for i in range(1, n_components+1)]
    df_pca = pd.DataFrame(X_pca, columns=pca_columns)
    df_pca['action'] = y.values
    
    # Speichern der PCA-Meta-Features
    pca_meta_filename = os.path.join(output_dir, 'pca_meta_features.pkl')
    df_pca.to_pickle(pca_meta_filename)
    print(f"PCA-Meta-Features gespeichert unter '{pca_meta_filename}'")
    
    return df_pca

# 3. Hauptfunktion
def main():
    # Pfad zur gespeicherten Pickle-Datei
    filepath = 'lunar_lander_data.pkl'
    
    # Daten laden
    df = load_data(filepath)
    if df is None:
        return
    
    # EDA durchführen (bereits vorhanden, hier nur als Referenz)
    # perform_eda(df)  # Stelle sicher, dass diese Funktion in deinem bestehenden Skript vorhanden ist
    
    # Feature Importance analysieren (bereits vorhanden, hier nur als Referenz)
    # importances = feature_importance_analysis(df)  # Stelle sicher, dass diese Funktion in deinem bestehenden Skript vorhanden ist
    
    # PCA durchführen und Loadings analysieren
    df_pca = perform_pca(df, n_components=5)
    
    # Optional: Weitere Schritte wie Feature Engineering oder Modelltraining können hier hinzugefügt werden

if __name__ == "__main__":
    main()
