import numpy as np
import os
from game import simulate_game
from game_hetero import simulate_heterogeneous_game
from imperfect_collusion_game import simulate_imperfect_collusion_game
from soft_risky_collusion_game import simulate_risky_collusion_game
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix

def extract_features(save_dir):
    rewards = np.load(os.path.join(save_dir, 'history_rewards.npy'))
    regrets = np.load(os.path.join(save_dir, 'history_regrets.npy'))
    avg_rewards = rewards.mean(axis=1)
    avg_regrets = regrets.mean(axis=1)
    features = np.vstack([avg_rewards, avg_regrets]).T
    return features

def true_labels(num_players, num_colluders):
    return np.array([1 if i < num_colluders else 0 for i in range(num_players)])

# === Main script ===

num_simulations = 50
train_size = 35

X_train, y_train = [], []
X_test, y_test = [], []

# Simulate multiple randomized games
for i in range(num_simulations):
    num_players = np.random.randint(6, 15)  # e.g., between 6 and 14 players
    num_colluders = np.random.randint(2, max(3, num_players // 2))  # at least 2, up to half

    sim_dir = f"game_data_{i}"
    simulate_risky_collusion_game(save_dir=sim_dir,
                  seed=i,
                  num_players=num_players,
                  num_colluders=num_colluders)

    X = extract_features(sim_dir)
    y = true_labels(num_players, num_colluders)

    if i < train_size:
        X_train.extend(X)
        y_train.extend(y)
    else:
        X_test.extend(X)
        y_test.extend(y)

X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

# === KMeans clustering ===
kmeans = KMeans(n_clusters=2, random_state=42)
train_clusters = kmeans.fit_predict(X_train)

def infer_cluster_mapping(cluster_labels, true_labels):
    map0 = np.mean(true_labels[cluster_labels == 0])
    map1 = np.mean(true_labels[cluster_labels == 1])
    return {0: int(map0 > map1), 1: int(map1 >= map0)}

mapping = infer_cluster_mapping(train_clusters, y_train)
test_clusters = kmeans.predict(X_test)
test_preds = np.array([mapping[c] for c in test_clusters])

# === Evaluation ===
acc = accuracy_score(y_test, test_preds)
conf = confusion_matrix(y_test, test_preds)

print("\n=== KMeans Collusion Detection on Randomized Games ===")
print("Accuracy:", round(acc, 3))
print("Confusion Matrix:\n", conf)
