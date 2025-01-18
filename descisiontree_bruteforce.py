# decisiontree_bruteforce.py

import gym
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

def collect_data(env, num_episodes=100, max_steps=200, policy='random'):
    """
    Sammelt Daten aus der Umgebung.

    :param env: OpenAI Gym Umgebung
    :param num_episodes: Anzahl der Episoden zum Sammeln von Daten
    :param max_steps: Maximale Schritte pro Episode
    :param policy: 'random' für zufällige Aktionen oder 'heuristic' für eine einfache Regel-basierte Politik
    :return: Zustände und Aktionen als NumPy-Arrays
    """
    states = []
    actions = []

    for episode in range(num_episodes):
        reset_result = env.reset()
        if isinstance(reset_result, tuple):
            state = reset_result[0]  # Gymnasium gibt (observation, info) zurück
        else:
            state = reset_result  # Gym gibt nur observation zurück
        for step in range(max_steps):
            if policy == 'random':
                action = env.action_space.sample()
            elif policy == 'heuristic':
                # Beispiel einer einfachen heuristischen Politik basierend auf Winkel und Winkelgeschwindigkeit
                angle = state[4]
                angular_vel = state[5]
                if angle < -0.1:
                    action = 1  # Feuer linken Booster
                elif angle > 0.1:
                    action = 3  # Feuer rechten Booster
                else:
                    action = 0  # Kein Feuer
            else:
                raise ValueError("Unbekannte Politik")

            states.append(state)
            actions.append(action)

            step_result = env.step(action)
            if len(step_result) == 5:
                next_state, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            elif len(step_result) == 4:
                next_state, reward, done, info = step_result
            else:
                raise ValueError(f"Unerwartete Anzahl von Rückgabewerten von env.step(): {len(step_result)}")

            state = next_state
            if done:
                break

    return np.array(states), np.array(actions)

def train_decision_trees(X_train, y_train, depth=5, num_trees=10):
    """
    Trainiert mehrere Entscheidungsbäume mit unterschiedlichen zufälligen Zuständen.

    :param X_train: Trainingszustände
    :param y_train: Trainingsaktionen
    :param depth: Maximale Tiefe der Bäume
    :param num_trees: Anzahl der zu trainierenden Bäume
    :return: Liste der trainierten Entscheidungsbäume
    """
    trees = []
    for i in range(num_trees):
        tree = DecisionTreeClassifier(max_depth=depth, random_state=i)
        tree.fit(X_train, y_train)
        trees.append(tree)
        print(f"Entscheidungsbaum {i+1}/{num_trees} trainiert.")
    return trees

def evaluate_tree(env, tree, num_episodes=20, max_steps=200):
    """
    Bewertet einen Entscheidungsbaum, indem er als Politik in der Umgebung verwendet wird.

    :param env: OpenAI Gym Umgebung
    :param tree: Entscheidungsbaum-Modell
    :param num_episodes: Anzahl der Episoden zur Bewertung
    :param max_steps: Maximale Schritte pro Episode
    :return: Durchschnittliche Belohnung
    """
    total_rewards = []

    for episode in range(num_episodes):
        reset_result = env.reset()
        if isinstance(reset_result, tuple):
            state = reset_result[0]  # Gymnasium gibt (observation, info) zurück
        else:
            state = reset_result  # Gym gibt nur observation zurück
        episode_reward = 0
        for step in range(max_steps):
            action = tree.predict([state])[0]
            step_result = env.step(action)
            if len(step_result) == 5:
                next_state, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            elif len(step_result) == 4:
                next_state, reward, done, info = step_result
            else:
                raise ValueError(f"Unerwartete Anzahl von Rückgabewerten von env.step(): {len(step_result)}")

            episode_reward += reward
            state = next_state
            if done:
                break
        total_rewards.append(episode_reward)

    mean_reward = np.mean(total_rewards)
    return mean_reward

def main():
    # Feature-Namen entsprechend der LunarLander-Beobachtungen
    feature_names = [
        'x_position',
        'y_position',
        'x_velocity',
        'y_velocity',
        'angle',
        'angular_velocity',
        'left_leg_contact',
        'right_leg_contact'
    ]

    # Umgebung initialisieren
    env = gym.make('LunarLander-v2')

    # Daten sammeln
    print("Daten werden gesammelt...")
    states, actions = collect_data(env, num_episodes=500, policy='random')
    print(f"Gesammelte Daten: {states.shape[0]} Zustände.")

    # Daten aufteilen
    X_train, X_test, y_train, y_test = train_test_split(states, actions, test_size=0.2, random_state=42)

    # Entscheidungsbäume trainieren
    print("Trainiere Entscheidungsbäume...")
    num_trees = 20  # Anzahl der verschiedenen Bäume
    trees = train_decision_trees(X_train, y_train, depth=5, num_trees=num_trees)

    # Bäume evaluieren
    print("Bewerte Entscheidungsbäume...")
    mean_rewards = []
    for i, tree in enumerate(trees):
        mean_reward = evaluate_tree(env, tree, num_episodes=30)
        mean_rewards.append(mean_reward)
        print(f"Baum {i+1}: Durchschnittliche Belohnung = {mean_reward:.2f}")

    # Ergebnisse plotten
    plt.figure(figsize=(12, 8))
    plt.bar(range(1, num_trees+1), mean_rewards, color='skyblue')
    plt.xlabel('Entscheidungsbaum', fontsize=14)
    plt.ylabel('Durchschnittliche Belohnung', fontsize=14)
    plt.title('Evaluation verschiedener Entscheidungsbäume für LunarLander', fontsize=16)
    plt.xticks(range(1, num_trees+1))
    plt.ylim([min(mean_rewards) - 10, max(mean_rewards) + 10])
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig('decision_trees_evaluation.png')
    plt.show()

    # Optional: Den besten Baum speichern
    best_index = np.argmax(mean_rewards)
    best_tree = trees[best_index]
    with open('best_decision_tree.pkl', 'wb') as f:
        pickle.dump(best_tree, f)
    print(f"Besten Baum (Baum {best_index+1}) wurde gespeichert.")

    # Optional: Den besten Baum als Text anzeigen mit korrekten Feature-Namen
    tree_rules = export_text(best_tree, feature_names=feature_names)
    print("\nBestes Entscheidungsbaum-Regelwerk:")
    print(tree_rules)

    # Optional: Genauigkeit des besten Baums auf dem Testdatensatz
    y_pred = best_tree.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nGenauigkeit des besten Baums auf dem Testdatensatz: {accuracy:.2f}")

    env.close()

if __name__ == "__main__":
    main()
