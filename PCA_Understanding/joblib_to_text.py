import gym
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier, export_text
import joblib
import os
import warnings
import time


def print_decision_tree_text(tree, feature_names):
    """
    Gibt den Entscheidungsbaum als Text aus.

    Args:
        tree: Das geladene DecisionTreeClassifier-Modell.
        feature_names: Die Namen der Features, die im Baum verwendet werden.
    """
    tree_rules = export_text(tree, feature_names=feature_names)
    print("----- Entscheidungsbaum -----")
    print(tree_rules)
    print("-----------------------------\n")



def main():
    MODEL_DIR = "decision_trees" 
    tree_path = os.path.join(MODEL_DIR, 'test_decision_tree.joblib')
    tree = joblib.load(tree_path)
    selected_features = joblib.load(os.path.join(MODEL_DIR, 'selected_features.joblib'))
    print_decision_tree_text(tree, selected_features)

if __name__ == "__main__":
    main()