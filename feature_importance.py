import joblib
import numpy as np
import matplotlib.pyplot as plt

# Load the model from the joblib file
model = joblib.load("decision_tree_models_gpt/run_3/trees/best_tree_depth_9.joblib")  # Replace with your actual file path

feature_names = [
            "x_space", "y_space", "vel_x_space", "vel_y_space",
            "angle", "angular_vel", "leg_1", "leg_2", "pc2", "pc4", "pc5", "speed",
            "vel_angle", "position_orientation_alignment", "position_heading_dot_product",
            "kinetic_energy", "rotational_kinetic_energy", "absolute_angular_vel",
            "angular_acceleration_estimate", "horizontal_instability_factor",
            "vertical_landing_readiness", "distance_to_center"
        ]

# Extract feature importances
importances = model.feature_importances_

# Sort features by importance
sorted_indices = np.argsort(importances)[::-1]
sorted_importances = importances[sorted_indices]
sorted_features = np.array(feature_names)[sorted_indices]

# Plot the feature importances
plt.figure(figsize=(10, 6))
plt.barh(sorted_features, sorted_importances, color="skyblue")
plt.xlabel("Feature Importance Score")
plt.ylabel("Features")
plt.title("Feature Importances from Decision Tree")
plt.gca().invert_yaxis()  # Invert y-axis to show the most important feature on top
plt.show()
