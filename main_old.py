from stable_baselines3 import PPO
from plot_utils_old import plot_model_detail_view, plot_model_overview

def main():
    # Lade das Modell
    model_path = "models\ppo_LunarLander-v2\ppo-LunarLander-v2.zip"
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
