import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd  # Neu: Für CSV-Export
import gymnasium as gym

def get_selected_action(model, state):
    """
    Gibt die vom Modell vorhergesagte Aktion für einen gegebenen Zustand zurück.
    """
    state = np.expand_dims(state, axis=0)  # Batch-Dimension hinzufügen
    action, _ = model.predict(state, deterministic=True)
    return action

def save_plot_data_and_generate(model, env_id="LunarLander-v3"):
    """
    Erstellt für jedes Paar von Zustandsdimensionen einen separaten Plot,
    speichert ihn in ./plots_detail_view/ und exportiert die Daten als CSV.
    """
    plot_dir = "./plots_detail_view"
    data_dir = "./plot_data"
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    # Zustandsräume definieren
    num_samples = 100
    observation_space = [
        np.linspace(-2.5, 2.5, num_samples),  # x_space
        np.linspace(-2.5, 2.5, num_samples),  # y_space
        np.linspace(-10, 10, num_samples),   # vel_x_space
        np.linspace(-10, 10, num_samples),   # vel_y_space
        np.linspace(-6.283, 6.283, num_samples),  # angle
        np.linspace(-10, 10, num_samples),   # angular_vel
        np.linspace(0, 1, 2),                # leg_1
        np.linspace(0, 1, 2)                 # leg_2
    ]
    labels = [
        "x_space", "y_space", "vel_x_space", "vel_y_space",
        "angle", "angular_vel", "leg_1", "leg_2"
    ]

    # Fixiere andere Zustandsdimensionen
    fixed_state = [0.0] * len(observation_space)

    for x_index in range(len(observation_space)):
        for y_index in range(x_index + 1, len(observation_space)):
            axis_0 = observation_space[x_index]
            axis_1 = observation_space[y_index]

            action_grid = np.zeros((len(axis_1), len(axis_0)))
            data_rows = []  # Daten für CSV

            for i, val_y in enumerate(axis_1):
                for j, val_x in enumerate(axis_0):
                    state = fixed_state.copy()
                    state[x_index] = val_x
                    state[y_index] = val_y
                    action = get_selected_action(model, state)
                    action_grid[i, j] = action
                    # Speichere Zustand und Aktion in die Datenreihe
                    data_rows.append([val_x, val_y, action])

            # Exportiere die Daten als CSV
            df = pd.DataFrame(data_rows, columns=[labels[x_index], labels[y_index], "action"])
            csv_filename = f"{labels[x_index]}_vs_{labels[y_index]}.csv"
            df.to_csv(os.path.join(data_dir, csv_filename), index=False)

            # Plot erstellen
            plt.figure(figsize=(10, 8))
            cax = plt.imshow(
                action_grid,
                extent=[axis_0.min(), axis_0.max(), axis_1.min(), axis_1.max()],
                origin='lower',
                cmap="viridis",
                aspect='auto'
            )
            cbar = plt.colorbar(cax, ticks=[0, 1, 2, 3])
            cbar.ax.set_yticklabels([
                "0: do nothing", "1: fire left engine",
                "2: fire main engine", "3: fire right engine"
            ])
            plt.title(f"{labels[x_index]} vs {labels[y_index]}")
            plt.xlabel(labels[x_index])
            plt.ylabel(labels[y_index])

            # Speichern
            pdf_filename = f"{labels[x_index]}_vs_{labels[y_index]}.pdf"
            plt.savefig(os.path.join(plot_dir, pdf_filename))
            plt.close()
