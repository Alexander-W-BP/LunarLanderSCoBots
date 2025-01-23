import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
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

# 2. Hinzufügen der Meta-Features zum DataFrame
def add_meta_features(df):
    """
    Fügt dem DataFrame die definierten Meta-Features hinzu.
    Verwendet die korrekten Spaltennamen:
    'x_position', 'y_position', 'x_velocity', 'y_velocity', 'angle', 'angular_velocity', 'left_leg_contact', 'right_leg_contact'

    Args:
        df (pd.DataFrame): Ursprünglicher Datensatz.

    Returns:
        pd.DataFrame: Datensatz mit zusätzlichen Meta-Features.
    """
    df['rotational_state'] = df['angle'] + 0.5 * df['angular_velocity']
    df['horizontal_state'] = df['x_position'] + 0.5 * df['x_velocity']
    df['vertical_state'] = df['y_position'] + 0.5 * df['y_velocity']
    df['leg_contact'] = df['left_leg_contact'] + df['right_leg_contact']
    print("Meta-Features wurden dem Datensatz hinzugefügt.")
    return df

# 3. Feature Importance Analyse
def feature_importance_analysis(df, output_dir='feature_importance_outputs'):
    """
    Bestimmt die Wichtigkeit der Features mittels Random Forest und visualisiert die Ergebnisse.

    Args:
        df (pd.DataFrame): Datensatz mit Zuständen, Aktionen und Meta-Features.
        output_dir (str): Verzeichnis zum Speichern der Feature-Importance-Ergebnisse.

    Returns:
        pd.Series: Sortierte Feature Importance.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("\n--- Feature Importance Analyse ---\n")

    # Zielvariable und Features
    X = df.drop('action', axis=1)
    y = df['action']

    # Aufteilen in Trainings- und Testdaten
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Random Forest Classifier
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    # Vorhersagen auf den Testdaten
    y_pred = rf.predict(X_test)

    # Evaluation
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Testgenauigkeit des Random Forest: {accuracy:.4f}")
    print("\nKlassifikationsbericht:")
    print(classification_report(y_test, y_pred))

    # Feature Importance extrahieren
    importances = pd.Series(rf.feature_importances_, index=X.columns)
    importances = importances.sort_values(ascending=False)
    importances.to_csv(os.path.join(output_dir, 'feature_importances.csv'))
    print("Feature Importance gespeichert unter 'feature_importances.csv'")

    # Visualisierung der Feature Importance
    plt.figure(figsize=(10, 8))
    sns.barplot(x=importances, y=importances.index, palette='viridis')
    plt.title("Feature Importance mittels Random Forest")
    plt.xlabel("Wichtigkeit")
    plt.ylabel("Feature")
    plt.savefig(os.path.join(output_dir, 'feature_importances.png'))
    plt.close()
    print("Plot 'feature_importances.png' gespeichert.")

    print("\nFeature Importance Analyse abgeschlossen und Ergebnisse gespeichert.")

    return importances

# 4. Speicherung der Feature Importance
def save_feature_importance(importances, filename='feature_importance.csv', output_dir='feature_importance_outputs'):
    """
    Speichert die Feature Importance in einer CSV-Datei.

    Args:
        importances (pd.Series): Sortierte Feature Importance.
        filename (str): Dateiname für die gespeicherte CSV.
        output_dir (str): Verzeichnis zum Speichern der Feature-Importance-Ergebnisse.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    importances.to_csv(os.path.join(output_dir, filename), header=['importance'])
    print(f"Feature Importance wurde in '{filename}' gespeichert.")

# 5. Hauptfunktion
def main():
    # Pfad zur gespeicherten Pickle-Datei
    filepath = 'lunar_lander_data.pkl' # Passe den Pfad an

    # Daten laden
    df = load_data(filepath)
    if df is None:
        return

    # Stelle sicher, dass die Spaltennamen korrekt sind
    print("Spaltennamen im DataFrame:", df.columns)

    # Meta-Features hinzufügen
    df = add_meta_features(df)

    # Feature Importance analysieren
    importances = feature_importance_analysis(df)

if __name__ == "__main__":
    main()