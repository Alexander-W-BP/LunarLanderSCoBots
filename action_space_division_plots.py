import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

def get_selected_action(model, state):
    """
    Gibt die vom Modell vorhergesagte Aktion für einen gegebenen Zustand zurück.
    """
    state = np.expand_dims(state, axis=0) 
    action, _ = model.predict(state, deterministic=True)
    return action

def plot_model_detail_view(model):
    """
    Erstellt für jedes Paar von Zustandsdimensionen einen separaten Plot,
    speichert ihn unter ./plots_detail_view_old/ und sammelt die Plot-Daten
    (action_grid, Achsenwerte, Labels) in einer Liste, die zurückgegeben wird.
    """
    plot_dir = "./plots_detail_view_old"
    os.makedirs(plot_dir, exist_ok=True)

    num_samples = 100

    observation_space = [
        np.linspace(-2.5, 2.5, num_samples),   # x
        np.linspace(-2.5, 2.5, num_samples),   # y
        np.linspace(-10, 10, num_samples),     # v_x
        np.linspace(-10, 10, num_samples),     # v_y
        np.linspace(-6.283, 6.283, num_samples),  # angle
        np.linspace(-10, 10, num_samples),     # v_angle
        np.linspace(0, 1, 2),                  # right_leg
        np.linspace(0, 1, 2),                  # left_leg
    ]
    labels = ["x", "y", "v_x", "v_y", "angle", "v_angle", "right_leg", "left_leg"]

    fixed_state = [0.0] * len(observation_space)

    detail_data = []

    for x_index in range(len(observation_space)):
        for y_index in range(x_index + 1, len(observation_space)):
            axis_0 = observation_space[x_index]
            axis_1 = observation_space[y_index]

            action_grid = np.zeros((len(axis_1), len(axis_0)))

            for i, val_y in enumerate(axis_1):
                for j, val_x in enumerate(axis_0):
                    state = fixed_state.copy()
                    state[x_index] = val_x
                    state[y_index] = val_y
                    action = get_selected_action(model, state)
                    action_grid[i, j] = action[0]  

            plt.figure(figsize=(10, 8))
            cax = plt.imshow(
                action_grid,
                extent=[axis_0.min(), axis_0.max(), axis_1.min(), axis_1.max()],
                origin="lower",
                cmap="viridis",
                aspect="auto",
            )
            cbar = plt.colorbar(cax, ticks=[0, 1, 2, 3])
            cbar.ax.set_yticklabels(
                [
                    "0: do nothing",
                    "1: fire left engine",
                    "2: fire main engine",
                    "3: fire right engine",
                ]
            )
            plt.title(f"{labels[x_index]} vs {labels[y_index]}")
            plt.xlabel(labels[x_index])
            plt.ylabel(labels[y_index])

            # Speichern
            pdf_filename = f"{labels[x_index]}_vs_{labels[y_index]}.pdf"
            plt.savefig(os.path.join(plot_dir, pdf_filename))
            plt.close()

            detail_data.append({
                "action_grid": action_grid,
                "x_vals": axis_0,
                "y_vals": axis_1,
                "x_label": labels[x_index],
                "y_label": labels[y_index],
            })

    return detail_data

def plot_model_overview(detail_data):
    """
    Nimmt die in plot_model_detail_view() bereits berechneten Plot-Daten
    und erzeugt daraus eine Übersicht mit allen Dimensionen-Paaren in
    EINER großen Figure (Mehrere Subplots).
    Speichert das Ergebnis als PDF unter ./plots_overview_old/overview_plots.pdf.

    Hier OHNE einzelne Colorbars neben jedem Subplot.
    """
    plot_dir = "./plots_overview_old"
    os.makedirs(plot_dir, exist_ok=True)

    # Anzahl der Plots = Anzahl der Einträge in detail_data
    n_plots = len(detail_data)

    # Spaltenzahl, Zeilenzahl dynamisch an n_plots anpassen
    cols = 3
    rows = (n_plots + cols - 1) // cols  # Aufrunden

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))
    axes = axes.flatten()

    for idx, data in enumerate(detail_data):
        ax = axes[idx]

        # Daten aus detail_data holen
        action_grid = data["action_grid"]
        x_vals = data["x_vals"]
        y_vals = data["y_vals"]
        x_label = data["x_label"]
        y_label = data["y_label"]

        # Plotten (Heatmap) - keine separate Colorbar
        im = ax.imshow(
            action_grid,
            extent=[x_vals.min(), x_vals.max(), y_vals.min(), y_vals.max()],
            origin="lower",
            cmap="viridis",
            aspect="auto",
        )
        ax.set_title(f"{x_label} vs {y_label}")
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

    # Falls mehr Subplots existieren als wir Daten haben, restliche löschen
    for j in range(n_plots, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    output_file = os.path.join(plot_dir, "overview_plots.pdf")
    plt.savefig(output_file)
    plt.close()
    print(f"Übersichtsplots erfolgreich gespeichert unter: {output_file}")

def main():
    model_path = "models\\ppo_LunarLander-v2\\ppo-LunarLander-v2.zip"
    model = PPO.load(model_path)

    print("Erstelle Detailplots...")
    detail_data = plot_model_detail_view(model)

    print("Erstelle Übersichtsplots...")
    plot_model_overview(detail_data)

    print("Plots erfolgreich erstellt!")

if __name__ == "__main__":
    main()
