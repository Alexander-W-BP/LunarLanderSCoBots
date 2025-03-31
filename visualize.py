import os
import re
import joblib
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# Basisverzeichnis, in dem alle Runs gespeichert wurden
BASE_DIR = "decision_tree_models_original"

# Finde den neuesten Run-Ordner (z.B. run_1, run_2, ...)
max_run_num = -1
latest_run_folder = None
for entry in os.listdir(BASE_DIR):
    entry_path = os.path.join(BASE_DIR, entry)
    if os.path.isdir(entry_path):
        match = re.match(r'run_(\d+)', entry)
        if match:
            run_num = int(match.group(1))
            if run_num > max_run_num:
                max_run_num = run_num
                latest_run_folder = entry

if latest_run_folder is None:
    print("Kein Run-Ordner gefunden.")
    exit(1)

# Annahme: Die Entscheidungsbäume wurden als joblib in einem Unterordner "trees" gespeichert.
trees_folder = os.path.join(BASE_DIR, latest_run_folder, "trees")
if not os.path.isdir(trees_folder):
    print(f"Kein Trees-Ordner gefunden im neuesten Run-Ordner: {latest_run_folder}.")
    exit(1)

# Definiere die Feature-Namen, wie sie auch beim Training genutzt wurden.
feature_names = ["x_space", "y_space", "vel_x_space", "vel_y_space", "angle", "angular_vel", "leg_1", "leg_2"]

# Definiere die Aktionen für die Blattknoten
action_names = ["do nothing", "left engine", "main engine", "right engine"]

# Erstelle einen Ausgabeordner für die Baumvisualisierungen
visualization_folder = os.path.join(BASE_DIR, latest_run_folder, "visualizations")
os.makedirs(visualization_folder, exist_ok=True)

# Iteriere über alle joblib-Dateien im Trees-Ordner
for filename in os.listdir(trees_folder):
    if filename.endswith(".joblib"):
        tree_path = os.path.join(trees_folder, filename)
        clf = joblib.load(tree_path)
        
        plt.figure(figsize=(12, 8))
        
        # plot_tree gibt eine Liste von Matplotlib-Textobjekten zurück
        text_objects = plot_tree(
            clf,
            feature_names=feature_names,
            class_names=action_names,
            filled=True,
            impurity=False,     # Gini/Entropie ausblenden
            proportion=False,   # keine prozentualen Anteile
            node_ids=False,     # keine Knotennummern
            rounded=True,
            precision=2         # z.B. zwei Nachkommastellen bei Schwellen
        )
        
        # Nachträglich die Zeilen entfernen, die mit "samples =" oder "value =" beginnen
        for t in text_objects:
            txt = t.get_text()
            lines = txt.split('\n')
            new_lines = []
            for line in lines:
                # Nur Zeilen behalten, die nicht mit "samples =" oder "value =" starten
                if not (line.startswith('samples =') or line.startswith('value =')):
                    new_lines.append(line)
            # Den bereinigten Text wieder setzen
            t.set_text('\n'.join(new_lines))
        
        plt.title(f"Entscheidungsbaum: {filename}")
        
        # Speichere die Visualisierung als PNG
        image_filename = os.path.join(visualization_folder, f"{os.path.splitext(filename)[0]}.png")
        plt.savefig(image_filename, bbox_inches="tight")
        plt.close()
        print(f"Baumvisualisierung gespeichert: {image_filename}")
