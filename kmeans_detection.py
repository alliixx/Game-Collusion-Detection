import numpy as np
import os
import joblib
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix
from collections import Counter
from game import simulate_game
from risky_collusion_game import simulate_risky_collusion_game
from auction_game import simulate_budgeted_auction_game

def extract_features(save_dir):
    rewards = np.load(os.path.join(save_dir, 'history_rewards.npy'))
    regrets = np.load(os.path.join(save_dir, 'history_regrets.npy'))
    avg_rewards = rewards.mean(axis=1)
    avg_regrets = regrets.mean(axis=1)
    return np.vstack([avg_rewards, avg_regrets]).T

def true_labels(num_players, num_colluders):
    return np.array([1 if i < num_colluders else 0 for i in range(num_players)])

def infer_cluster_mapping(cluster_labels, true_labels):
    map0 = np.mean(true_labels[cluster_labels == 0])
    map1 = np.mean(true_labels[cluster_labels == 1])
    return {0: int(map0 > map1), 1: int(map1 >= map0)}

# === Config ===
train_simulations = 35
test_simulations = 15
num_players_range = (6, 15)

game_types = {
    "risky": simulate_risky_collusion_game,
    "normal": simulate_game,
    "auction": simulate_budgeted_auction_game
}

for game_name, simulate_fn in game_types.items():
    print(f"\n=== Training and Evaluating KMeans on {game_name.upper()} Games ===")
    
    X_train, y_train, X_test, y_test = [], [], [], []

    # --- Training ---
    for i in range(train_simulations):
        num_players = np.random.randint(*num_players_range)
        num_colluders = np.random.randint(0, max(3, num_players // 2))
        sim_dir = f"kmeans_{game_name}_train_{i}"
        simulate_fn(save_dir=sim_dir, num_players=num_players, num_colluders=num_colluders)
        X_train.extend(extract_features(sim_dir))
        y_train.extend(true_labels(num_players, num_colluders))

    # --- Testing ---
    for i in range(test_simulations):
        num_players = np.random.randint(*num_players_range)
        num_colluders = np.random.randint(0, max(3, num_players // 2))
        sim_dir = f"kmeans_{game_name}_test_{i}"
        simulate_fn(save_dir=sim_dir, num_players=num_players, num_colluders=num_colluders)
        X_test.extend(extract_features(sim_dir))
        y_test.extend(true_labels(num_players, num_colluders))

    # --- Convert to numpy ---
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_test, y_test = np.array(X_test), np.array(y_test)

    # --- Train KMeans ---
    kmeans = KMeans(n_clusters=2, random_state=42)
    train_clusters = kmeans.fit_predict(X_train)
    mapping = infer_cluster_mapping(train_clusters, y_train)
    test_preds = np.array([mapping[c] for c in kmeans.predict(X_test)])

    # --- Evaluation ---
    print("Accuracy:", round(accuracy_score(y_test, test_preds), 3))
    print("Confusion Matrix:\n", confusion_matrix(y_test, test_preds))

    # --- Save Model & Mapping ---
    joblib.dump(kmeans, f"kmeans_{game_name}_model.pkl")
    joblib.dump(mapping, f"kmeans_{game_name}_mapping.pkl")
    print(f"Saved: kmeans_{game_name}_model.pkl and kmeans_{game_name}_mapping.pkl")
