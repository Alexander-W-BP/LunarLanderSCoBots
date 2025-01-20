import os
import glob
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Muss vor import matplotlib.pyplot stehen!
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def plot_decision_boundaries(
    tree, X, y, feature_names, filename="vergleich_plot.png"
):
    """
    Erzeugt zwei Subplots untereinander:
    (1) Decision-Boundaries + Entscheidungs-Linien
    (2) Originaldaten-Punkte (Scatter)
    mit dynamischer Colorbar gemäß den tatsächlich vorkommenden Klassen.
    """

    # Grid vorbereiten
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 500),
        np.linspace(y_min, y_max, 500)
    )
    grid = np.c_[xx.ravel(), yy.ravel()]

    Z = tree.predict(grid).reshape(xx.shape)

    # Welche Klassen kommen überhaupt vor?
    unique_classes = sorted(np.unique(y))

    # Für diskrete Farbabstufung legen wir Levels an:
    # z.B. classes {0,1,2} => levels = [-0.5, 0.5, 1.5, 2.5]
    # classes {0,1,2,3} => [-0.5,0.5,1.5,2.5,3.5] usw.
    levels = [uc - 0.5 for uc in unique_classes] + [unique_classes[-1] + 0.5]

    fig, axs = plt.subplots(nrows=2, figsize=(8, 12))

    # -------------------- (1) Oberer Plot: Decision-Boundaries --------------------
    ax1 = axs[0]
    contour = ax1.contourf(xx, yy, Z, levels=levels, cmap="viridis", alpha=0.6)
    # Colorbar mit dynamischen Ticks (exactly the unique classes)
    cbar1 = fig.colorbar(contour, ax=ax1, ticks=unique_classes)

    # Beispielhafte Beschriftungen für Aktion 0..3:
    action_labels_map = {
        0: "0: do nothing",
        1: "1: fire left engine",
        2: "2: fire main engine",
        3: "3: fire right engine"
    }
    # Falls Klassen > 3 oder Lücken, fallback generisch "class X"
    cbar1.ax.set_yticklabels([action_labels_map.get(c, f"class {c}")
                              for c in unique_classes])

    # Linien für Entscheidungsregeln
    rules_lines = export_text(tree, feature_names=feature_names).split("\n")
    for rule_line in rules_lines:
        if "class:" in rule_line or not rule_line.strip():
            continue

        cleaned_line = (rule_line
                        .replace("|--- ", "")
                        .replace("|   ", "")
                        .strip())

        if "<=" in cleaned_line:
            feat, thr = cleaned_line.split("<=")
            feat, thr = feat.strip(), float(thr.strip())
            if feat == feature_names[0]:
                ax1.axvline(x=thr, color="red", linestyle="--",
                            label=f"{feat} <= {thr:.2f}")
            elif feat == feature_names[1]:
                ax1.axhline(y=thr, color="blue", linestyle="--",
                            label=f"{feat} <= {thr:.2f}")
        elif ">" in cleaned_line:
            feat, thr = cleaned_line.split(">")
            feat, thr = feat.strip(), float(thr.strip())
            if feat == feature_names[0]:
                ax1.axvline(x=thr, color="red", linestyle=":",
                            label=f"{feat} > {thr:.2f}")
            elif feat == feature_names[1]:
                ax1.axhline(y=thr, color="blue", linestyle=":",
                            label=f"{feat} > {thr:.2f}")

    ax1.set_xlabel(feature_names[0])
    ax1.set_ylabel(feature_names[1])
    ax1.set_title("1) Entscheidungsgrenzen des Decision Trees")
    # Entferne doppelte Legendeneinträge
    handles, labels = ax1.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax1.legend(unique.values(), unique.keys(), loc="upper right")

    # -------------------- (2) Unterer Plot: Originaldaten-Punkte --------------------
    ax2 = axs[1]
    scatter = ax2.scatter(
        X[:, 0], X[:, 1],
        c=y,
        cmap="viridis",
        edgecolor='k',
        s=30,
        vmin=unique_classes[0],
        vmax=unique_classes[-1]  # min..max
    )
    cbar2 = fig.colorbar(scatter, ax=ax2, ticks=unique_classes)
    cbar2.ax.set_yticklabels([action_labels_map.get(c, f"class {c}")
                              for c in unique_classes])

    ax2.set_xlabel(feature_names[0])
    ax2.set_ylabel(feature_names[1])
    ax2.set_title("2) Originaldaten (Scatter)")

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def main():
    import glob

    # Zielordner für Grafiken und gemeinsame Regel-Datei
    os.makedirs("plot_cuts", exist_ok=True)
    all_rules_path = os.path.join("plot_cuts", "all_decision_rules.txt")

    # Öffne die Regeldatei EINMAL im Schreibmodus, leere sie ggf. vorher
    with open(all_rules_path, "w", encoding="utf-8") as f_rules:

        # Alle CSVs in plot_data durchgehen
        csv_files = glob.glob("plot_data/*.csv")
        for csv_file in csv_files:
            basename = os.path.basename(csv_file)  # z.B. "y_space_vs_vel_x_space.csv"
            base_no_ext = os.path.splitext(basename)[0]  # "y_space_vs_vel_x_space"
            png_filename = os.path.join("plot_cuts", f"{base_no_ext}.png")

            # Feature-Namen extrahieren, sofern "_vs_" im Dateinamen
            if "_vs_" in base_no_ext:
                f1, f2 = base_no_ext.split("_vs_")
                feature_names = [f1, f2]
            else:
                # Fallback: Oder du definierst manuell z.B.:
                feature_names = ["feature1", "feature2"]

            # CSV laden
            df = pd.read_csv(csv_file)
            if "action" not in df.columns:
                print(f"Fehler: 'action'-Spalte nicht in {csv_file} gefunden.")
                continue

            # Action bereinigen (Klammern entfernen) und in int konvertieren
            df["action"] = df["action"].astype(str).str.strip("[]")
            y = df["action"].astype(int).values

            # X holen
            if all(col in df.columns for col in feature_names):
                X = df[feature_names].values
            else:
                print(f"Spalten {feature_names} nicht in {csv_file}. Überspringe.")
                continue

            # Daten in Trainings- und Validierungsset aufteilen
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            # Initialisiere Variablen für die beste Modellwahl
            best_depth = 1
            best_acc = 0
            desired_acc = 0.95  # Beispiel: gewünschte Genauigkeit von 95%
            max_allowed_depth = 3  # Maximale Tiefe auf 3 beschränkt

            for depth in range(1, max_allowed_depth + 1):
                model = DecisionTreeClassifier(random_state=42, max_depth=depth)
                model.fit(X_train, y_train)

                y_val_pred = model.predict(X_val)
                acc = accuracy_score(y_val, y_val_pred)

                if acc > best_acc:
                    best_acc = acc
                    best_depth = depth

                # Stoppt, wenn die gewünschte Genauigkeit erreicht ist
                if acc >= desired_acc:
                    break

            # Trainiere das endgültige Modell mit der besten Tiefe auf allen Daten
            final_model = DecisionTreeClassifier(random_state=42, max_depth=best_depth)
            final_model.fit(X, y)

            # Entscheidungsregeln
            rules_text = export_text(final_model, feature_names=feature_names)

            # --> An EINEM Stück in die Sammel-Datei schreiben
            f_rules.write(f"=== Regeln für {basename} (Tiefe: {best_depth}) ===\n")
            f_rules.write(rules_text)
            f_rules.write("\n\n")

            # Plot erstellen
            plot_decision_boundaries(final_model, X, y, feature_names, filename=png_filename)

            # (Optional) Baumdiagramm des DecisionTrees speichern
            plt.figure(figsize=(12, 8))
            plot_tree(
                final_model,
                filled=True,
                feature_names=feature_names,
                class_names=[str(c) for c in sorted(np.unique(y))],
                rounded=True
            )
            tree_path = os.path.join("plot_cuts", f"tree_{base_no_ext}.png")
            plt.savefig(tree_path)
            plt.close()

            # (Optional) Genauigkeit am Trainingsdatensatz
            y_pred = final_model.predict(X)
            acc_train = accuracy_score(y, y_pred)

            print(f"Datei: {csv_file}")
            print(f"  Tiefe des Baumes: {best_depth}")
            print(f"  Trainingsgenauigkeit: {acc_train:.3f}")
            print(f"  PNG: {png_filename}")
            print(f"  Regeln in: {all_rules_path}")
            print("")


if __name__ == "__main__":
    main()
