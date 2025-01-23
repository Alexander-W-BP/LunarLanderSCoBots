from stable_baselines3 import PPO
from plot_utils import save_plot_data_and_generate

def main():
    # Lade das Modell
    model_path = "models/ppo-LunarLander-v3/best_model.zip"
    model = PPO.load(model_path)

    # Detailplots
    print("Erstelle Detailplots...")
    save_plot_data_and_generate(model)

    # Übersichtsplots
    #print("Erstelle Übersichtsplots...")
    #plot_model_overview(model)

    print("Plots erfolgreich erstellt!")

if __name__ == "__main__":
    main()
