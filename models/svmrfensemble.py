
import numpy as np
import os
import tempfile
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
import pandas as pd

from game import simulate_game
from risky_collusion_game import simulate_risky_collusion_game
from auction_game import simulate_budgeted_auction_game

def run_single_game(sim_func, num_players, colluding_ids, num_rounds):
    with tempfile.TemporaryDirectory() as tmpdirname:
        sim_func(num_players=num_players, num_colluders=len(colluding_ids),
                 save_dir=tmpdirname, num_rounds=num_rounds)
        rewards = np.load(os.path.join(tmpdirname, "history_rewards.npy"))
        regrets = np.load(os.path.join(tmpdirname, "history_regrets.npy"))
    labels = np.array([1 if i in colluding_ids else 0 for i in range(num_players)])
    return rewards, regrets, labels

def run_single_auction(sim_func, num_players, colluding_ids, num_items):
    with tempfile.TemporaryDirectory() as tmpdirname:
        sim_func(save_dir=tmpdirname, num_players=num_players, num_colluders=len(colluding_ids), num_items=num_items)
        rewards = np.load(os.path.join(tmpdirname, "history_rewards.npy"))
        regrets = np.load(os.path.join(tmpdirname, "history_regrets.npy"))
    labels = np.array([1 if i in colluding_ids else 0 for i in range(num_players)])
    return rewards, regrets, labels

def generate_dataset(sim_func, sim_type, num_sims=30, num_rounds=300):
    X, y = [], []
    for _ in range(num_sims):
        num_players = np.random.randint(4, 21)
        max_colluders = max(3, num_players // 2)
        colluding_ids = np.random.choice(num_players, size=np.random.randint(1, max_colluders + 1), replace=False).tolist()


        if sim_type == "auction":
            rewards, regrets, labels = run_single_auction(sim_func, num_players, colluding_ids, num_rounds)
        else:
            rewards, regrets, labels = run_single_game(sim_func, num_players, colluding_ids, num_rounds)

        for i in range(num_players):
            feat = [
                np.mean(rewards[i]), np.std(rewards[i]),
                np.mean(regrets[i]), np.std(regrets[i])
            ]
            X.append(feat)
            y.append(labels[i])
    return np.array(X), np.array(y)

def train_rf_svm_ensemble(X, y):
    rf = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=100, random_state=42))
    svm = make_pipeline(StandardScaler(), SVC(kernel='rbf', probability=True))

    rf.fit(X, y)
    svm.fit(X, y)

    def ensemble_predict(X):
        rf_probs = rf.predict_proba(X)[:, 1]
        svm_probs = svm.predict_proba(X)[:, 1]
        avg_probs = (rf_probs + svm_probs) / 2
        return (avg_probs >= 0.5).astype(int)

    return ensemble_predict

def plot_f1_over_time(env_results, label="F1 Score", window=5, save_path="f1_score_over_time.png"):
    plt.figure(figsize=(10, 6))
    for name, (X, y) in env_results.items():
        preds = ensemble_predict_fn(X)
        f1s = []
        chunk_size = 10
        for i in range(chunk_size, len(y) + 1, chunk_size):
            f1 = f1_score(y[:i], preds[:i])
            f1s.append(f1)
        smoothed = uniform_filter1d(f1s, size=window)
        plt.plot(range(chunk_size, len(y) + 1, chunk_size), smoothed, label=name)

    plt.title("Smoothed F1 Score Over Time")
    plt.xlabel("Examples")
    plt.ylabel(label)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# run pipeline
X_homo, y_homo = generate_dataset(simulate_game, "homo", num_sims=600)
X_train_homo, y_train_homo = X_homo[:500], y_homo[:500]
X_test_homo, y_test_homo = X_homo[500:], y_homo[500:]

X_hetero, y_hetero = generate_dataset(simulate_risky_collusion_game, "hetero", num_sims=600)
X_train_hetero, y_train_hetero = X_hetero[:500], y_hetero[:500]
X_test_hetero, y_test_hetero = X_hetero[500:], y_hetero[500:]

X_auction, y_auction = generate_dataset(simulate_budgeted_auction_game, "auction", num_sims=600)
X_train_auction, y_train_auction = X_auction[:500], y_auction[:500]
X_test_auction, y_test_auction = X_auction[500:], y_auction[500:]

X_all_train = np.vstack([X_train_homo, X_train_hetero, X_train_auction])
y_all_train = np.concatenate([y_train_homo, y_train_hetero, y_train_auction])

ensemble_predict_fn = train_rf_svm_ensemble(X_all_train, y_all_train)

env_data = {
    "Homogeneous": (X_test_homo, y_test_homo),
    "Heterogeneous": (X_test_hetero, y_test_hetero),
    "Auction": (X_test_auction, y_test_auction)
}

results = {}
predictions = []
for name, (X, y) in env_data.items():
    preds = ensemble_predict_fn(X)
    acc = accuracy_score(y, preds)
    f1 = f1_score(y, preds)
    cm = confusion_matrix(y, preds)
    results[name] = {"accuracy": acc, "f1_score": f1, "confusion_matrix": cm}
    for true, pred in zip(y, preds):
        predictions.append({"Environment": name, "True Label": true, "Predicted Label": pred})

# save csv + plot
pd.DataFrame(predictions).to_csv("per_agent_predictions.csv", index=False)
plot_f1_over_time(env_data, save_path="f1_score_over_time.png")
