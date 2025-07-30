import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
from games.risky_collusion_game import simulate_risky_collusion_game

def true_labels(num_players, num_colluders):
    return np.array([1 if i < num_colluders else 0 for i in range(num_players)])

def live_kmeans_predict(model, mapping, rewards, regrets, threshold_distance=0.5, patience=5):
    num_players, num_rounds = rewards.shape
    final_preds = np.full(num_players, -1)
    final_dists = np.full(num_players, np.inf)
    prediction_time = np.full(num_players, -1)
    cluster_history = [[] for _ in range(num_players)]

    for t in range(10, num_rounds):
        avg_rewards = rewards[:, :t+1].mean(axis=1)
        avg_regrets = regrets[:, :t+1].mean(axis=1)
        X_t = np.vstack([avg_rewards, avg_regrets]).T

        cluster_labels = model.predict(X_t)
        dists = model.transform(X_t)  # distances to centroids

        for i in range(num_players):
            cluster = cluster_labels[i]
            dist = dists[i][cluster]
            cluster_history[i].append(cluster)

            if (
                final_preds[i] == -1 and
                dist <= threshold_distance and
                len(cluster_history[i]) >= patience and
                all(x == cluster for x in cluster_history[i][-patience:])
            ):
                final_preds[i] = mapping[cluster]
                final_dists[i] = dist
                prediction_time[i] = t

        if np.all(final_preds != -1):
            break

    return final_preds, final_dists, prediction_time

# === Load saved model and mapping ===
kmeans = joblib.load("kmeans_model.pkl")
mapping = joblib.load("kmeans_mapping.pkl")

# === Run a new live game ===
live_dir = "live_kmeans_game"
num_players = 12
num_colluders = 3
simulate_risky_collusion_game(save_dir=live_dir, seed=42, num_players=num_players, num_colluders=num_colluders)

rewards = np.load(os.path.join(live_dir, 'history_rewards.npy'))
regrets = np.load(os.path.join(live_dir, 'history_regrets.npy'))
true = true_labels(num_players, num_colluders)

# === Predict ===
preds, dists, times = live_kmeans_predict(kmeans, mapping, rewards, regrets)

# === Output ===
print("\n=== Final Predictions (KMeans) ===")
print("Predicted colluders:", np.where(preds == 1)[0])
print("True colluders:     ", np.where(true == 1)[0])
print("Prediction times (rounds):", times.astype(int))

# === Plot ===
plt.figure(figsize=(10, 6))
plt.bar(range(num_players), 1 - dists, color=["red" if p == 1 else "blue" for p in preds])
plt.axhline(1 - 0.5, linestyle='--', color='gray', label="Distance Threshold")
plt.title("Final Cluster-Based Collusion Prediction Confidence")
plt.xlabel("Player ID")
plt.ylabel("1 - Distance to Cluster Center (Higher = More Confident)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
