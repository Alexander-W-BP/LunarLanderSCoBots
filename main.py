from stable_baselines3 import PPO
from plot_utils import plot_model_detail_view, plot_model_overview

def main():
    # Lade das Modell
    model_path = "models/ppo-LunarLander-v3/best_model.zip"
    model = PPO.load(model_path)

    # Detailplots
    print("Erstelle Detailplots...")
    plot_model_detail_view(model)

    # Übersichtsplots
    print("Erstelle Übersichtsplots...")
    plot_model_overview(model)

    print("Plots erfolgreich erstellt!")

if __name__ == "__main__":
    main()
