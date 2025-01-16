# Imports
import os
import numpy as np
from stable_baselines3 import DQN, PPO
import gymnasium as gym
import matplotlib.pyplot as plt

def load_model(algorithm, model_dir):
    """
    Lädt ein trainiertes Modell basierend auf dem Algorithmus.

    :param algorithm: Name des Algorithmus (z.B. 'DQN', 'PPO').
    :param model_dir: Pfad zum Modell.
    :return: Geladenes Modell.
    """
    if algorithm == "DQN":
        model = DQN.load(os.path.join(model_dir, "dqn_lunar_lander"))
    elif algorithm == "PPO":
        model = PPO.load(os.path.join(model_dir, "ppo_lunar_lander"))
    else:
        raise ValueError(f"Algorithmus '{algorithm}' wird nicht unterstützt.")
    return model

def plot_model_detail_view(model):
    # Define the LunarLander environment
    env = gym.make("LunarLander-v3")

    # Function to get the model's selected action for a given state
    def get_selected_action(model, state):
        state = np.expand_dims(state, axis=0)  # Add batch dimension
        action, _ = model.predict(state, deterministic=True)  # Get the action
        return action

    # define observation space with values from gymnasium documentation
    number_of_samples = 100
    x_space = np.linspace(-2.5, 2.5, number_of_samples)
    y_space = np.linspace(-2.5, 2.5, number_of_samples)
    vel_x_space = np.linspace(-10, 10, number_of_samples)
    vel_y_space = np.linspace(-10, 10, number_of_samples)
    angle = np.linspace(-6.2831855, 6.2831855, number_of_samples)
    angular_vel = np.linspace(-10, 10, number_of_samples)
    leg_1 = np.linspace(0, 1, 2)
    leg_2 = np.linspace(0, 1, 2)

    observation_space = [
        x_space, 
        y_space,
        vel_x_space,
        vel_y_space,
        angle,
        angular_vel,
        leg_1,
        leg_2
    ]

    labels = [
        "x_space",
        "y_space",
        "vel_x_space",
        "vel_y_space",
        "angle",
        "angular_vel",
        "leg_1",
        "leg_2"
    ]

    # Fixed values for other state dimensions
    fixed_state = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # [x, y, vel_x, vel_y, angle, angular_vel, leg1, leg2]

    for x_index in range(0, len(observation_space)):
      for y_index in range(x_index + 1, len(observation_space)):

        axis_0 = observation_space[x_index]
        axis_1 = observation_space[y_index]

        # Store the selected actions for visualization
        action_grid = np.zeros((len(axis_1), len(axis_0)))

        for i, y in enumerate(axis_1):
            for j, x in enumerate(axis_0):
                state = fixed_state.copy()
                # vary current features
                state[x_index] = x
                state[y_index] = y

                action = get_selected_action(model, state)
                action_grid[i, j] = action  # Store the selected action

        # Plot the action grid
        plt.figure(figsize=(10, 8))
        cax = plt.imshow(action_grid, extent=[axis_0.min(), axis_0.max(), axis_1.min(), axis_1.max()],
                   origin='lower', cmap="viridis", aspect='auto')
        cbar = plt.colorbar(cax, ticks=[0, 1, 2, 3])
        cbar.set_ticks([0, 1, 2, 3])
        cbar.set_ticklabels(["0: do nothing", "1: fire left orientation engine", "2: fire main engine", "3: fire right orientation engine"])
        plt.title("Actions Selected by Agent (LunarLander-v3)")
        plt.xlabel(labels[x_index])
        plt.ylabel(labels[y_index])
        plt.show()

def plot_model_overview(model):
    # Define the LunarLander environment
    env = gym.make("LunarLander-v3")

    # Function to get the model's selected action for a given state
    def get_selected_action(model, state):
        state = np.expand_dims(state, axis=0)  # Add batch dimension
        action, _ = model.predict(state, deterministic=True)  # Get the action
        return action

    # Define observation space with values from gym documentation
    number_of_samples = 100
    x_space = np.linspace(-2.5, 2.5, number_of_samples)
    y_space = np.linspace(-2.5, 2.5, number_of_samples)
    vel_x_space = np.linspace(-10, 10, number_of_samples)
    vel_y_space = np.linspace(-10, 10, number_of_samples)
    angle = np.linspace(-6.2831855, 6.2831855, number_of_samples)
    angular_vel = np.linspace(-10, 10, number_of_samples)
    leg_1 = np.linspace(0, 1, 2)
    leg_2 = np.linspace(0, 1, 2)

    observation_space = [
        x_space, 
        y_space,
        vel_x_space,
        vel_y_space,
        angle,
        angular_vel,
        leg_1,
        leg_2
    ]

    labels = [
        "x_space",
        "y_space",
        "vel_x_space",
        "vel_y_space",
        "angle",
        "angular_vel",
        "leg_1",
        "leg_2"
    ]

    # Fixed values for other state dimensions
    fixed_state = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # [x, y, vel_x, vel_y, angle, angular_vel, leg1, leg2]

    # Calculate the number of subplots required
    num_pairs = sum(1 for x_index in range(len(observation_space)) for y_index in range(x_index + 1, len(observation_space)))
    cols = 3
    rows = (num_pairs + cols - 1) // cols  # Calculate rows to fit all pairs

    # Adjust figure size to ensure square subplots
    subplot_size = 5  # Size of each subplot (square)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * subplot_size, rows * subplot_size))
    axes = axes.flatten()  # Flatten the axes array for easy iteration

    subplot_index = 0
    for x_index in range(0, len(observation_space)):
        for y_index in range(x_index + 1, len(observation_space)):
            axis_0 = observation_space[x_index]
            axis_1 = observation_space[y_index]

            # Store the selected actions for visualization
            action_grid = np.zeros((len(axis_1), len(axis_0)))

            for i, y in enumerate(axis_1):
                for j, x in enumerate(axis_0):
                    state = fixed_state.copy()
                    # Vary current features
                    state[x_index] = x
                    state[y_index] = y

                    action = get_selected_action(model, state)
                    action_grid[i, j] = action  # Store the selected action

            # Plot the action grid in the current subplot
            ax = axes[subplot_index]
            cax = ax.imshow(action_grid, extent=[axis_0.min(), axis_0.max(), axis_1.min(), axis_1.max()],
                            origin='lower', cmap="viridis", aspect='auto')
            cbar = fig.colorbar(cax, ax=ax, ticks=[0, 1, 2, 3])
            cbar.set_ticks([0, 1, 2, 3])
            # cbar.set_ticklabels(["0: do nothing", "1: fire left orientation engine", "2: fire main engine", "3: fire right orientation engine"])
            ax.set_title(f"{labels[x_index]} vs {labels[y_index]}")
            ax.set_xlabel(labels[x_index])
            ax.set_ylabel(labels[y_index])

            subplot_index += 1

    # Hide unused subplots if any
    for idx in range(subplot_index, len(axes)):
        axes[idx].axis('off')

    # Adjust layout and spacing
    plt.tight_layout(pad=3.0, w_pad=2.0, h_pad=3.0)
    plt.show()

# Hauptprogramm
def main():
    # Einstellungen
    algorithm = "DQN"  # Oder "PPO"
    model_dir = "./logs/DQN_optimized"  # Oder "./logs/PPO"
    #model_dir = "./logs/DQN/log_v4"

    # Modell laden
    print(f"Lade {algorithm}-Modell...")
    model = load_model(algorithm, model_dir)

    plot_model_detail_view(model)
    #plot_model_overview(model)

if __name__ == "__main__":
    main()
