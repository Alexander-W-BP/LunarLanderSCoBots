import pandas as pd
import matplotlib.pyplot as plt

# List of CSV filenames
csv_files = [
    'D:/Studium/Master/Semester3/KI_Praktikum/LunarLanderSCoBots/decision_tree_models_experiments_original_features/run_1/performance.csv',
    'D:/Studium/Master/Semester3/KI_Praktikum/LunarLanderSCoBots/decision_tree_models_experiments_chat_gpt_features/run_1/performance.csv',
    'D:/Studium/Master/Semester3/KI_Praktikum/LunarLanderSCoBots/performance.csv',
    'D:/Studium/Master/Semester3/KI_Praktikum/LunarLanderSCoBots/decision_tree_models_experiments_all_features/run_1/performance.csv',
    'D:/Studium/Master/Semester3/KI_Praktikum/LunarLanderSCoBots/decision_tree_models_experiments_pca_features/run_1/performance.csv',
    'D:/Studium/Master/Semester3/KI_Praktikum/LunarLanderSCoBots/decision_tree_models_experiments_top_5_features_only/run_1/performance.csv'
]

# Optional: Labels for each line on the plot
labels = [
    'Original Features',
    'LLM Features',
    'Only PCA',
    'Our Solution (Meta Features)',
    'PCA Features',
    'Top 5 Features'
]

# Plotting
plt.figure(figsize=(10, 6))

for i, file in enumerate(csv_files):
    data = pd.read_csv(file)
    depths = data['depths']
    mean_rewards = data['mean_rewards']
    std_rewards = data['std_rewards']  # Assuming this is the column name

    # Plot the mean line
    if (i == 44):
        plt.plot(depths, mean_rewards, label=labels[i], color='red')
        plt.fill_between(depths, mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=0.2, color='red')
    else:
        plt.plot(depths, mean_rewards, label=labels[i])
        #plt.fill_between(depths, mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=0.2)
    
    # Fill between mean ± std for shading
    #plt.errorbar(depths, mean_rewards, yerr=std_rewards, label=labels[i], capsize=3, marker='o', linestyle='-')
    #plt.fill_between(depths, mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=0.2)


plt.xlabel('Tree Depth')
plt.ylabel('Mean Reward (over all seeds and episodes)')
plt.title('Decision Tree: Mean Reward vs. Max Depth (LunarLander-v2)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
