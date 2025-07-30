import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
from games.game import simulate_game
from games.auction_game import simulate_budgeted_auction_game
from games.risky_collusion_game import simulate_risky_collusion_game
from scipy.ndimage import gaussian_filter1d

def true_labels(num_players, num_colluders):
    return np.array([1 if i < num_colluders else 0 for i in range(num_players)])

def extract_features_up_to_t(rewards, regrets, t):
    avg_rewards = rewards[:, :t+1].mean(axis=1)
    avg_regrets = regrets[:, :t+1].mean(axis=1)
    std_rewards = rewards[:, :t+1].std(axis=1)
    std_regrets = regrets[:, :t+1].std(axis=1)
    regret_over_reward = avg_regrets / (avg_rewards + 1e-6)
    return np.vstack([avg_rewards, avg_regrets, std_rewards, std_regrets, regret_over_reward]).T

def run_live_predictions(model_name, simulate_fn):
    print(f"\n=== Evaluating {model_name.upper()} Model ===")

    model = joblib.load(f"{model_name}_xgb_model.pkl")
    scaler = joblib.load(f"{model_name}_xgb_scaler.pkl")

    save_dir = f"live_{model_name}_game"
    os.makedirs(save_dir, exist_ok=True)
    num_players = 12
    num_colluders = 3
    simulate_fn(save_dir=save_dir, num_players=num_players, num_colluders=num_colluders)

    rewards = np.load(os.path.join(save_dir, 'history_rewards.npy'))
    regrets = np.load(os.path.join(save_dir, 'history_regrets.npy'))
    true = true_labels(num_players, num_colluders)
    num_rounds = rewards.shape[1]

    precision_list, recall_list, plot_times = [], [], []

    for t in range(1, num_rounds):  # Avoid division by zero at t=0
        X_t = extract_features_up_to_t(rewards, regrets, t)
        X_scaled = scaler.transform(X_t)
        preds = model.predict(X_scaled)

        pred_colluders = set(np.where(preds == 1)[0])
        true_colluders = set(np.where(true == 1)[0])

        tp = len(true_colluders & pred_colluders)
        fp = len(pred_colluders - true_colluders)
        fn = len(true_colluders - pred_colluders)

        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)

        precision_list.append(precision)
        recall_list.append(recall)
        plot_times.append(t)

    return plot_times, precision_list, recall_list

from sklearn.metrics import f1_score

def run_live_f1(model_name, simulate_fn):
    print(f"\n=== Evaluating {model_name.upper()} Model ===")

    model = joblib.load(f"{model_name}_xgb_model.pkl")
    scaler = joblib.load(f"{model_name}_xgb_scaler.pkl")

    save_dir = f"live_{model_name}_game"
    os.makedirs(save_dir, exist_ok=True)
    num_players = 12
    num_colluders = 3
    simulate_fn(save_dir=save_dir, num_players=num_players, num_colluders=num_colluders)

    rewards = np.load(os.path.join(save_dir, 'history_rewards.npy'))
    regrets = np.load(os.path.join(save_dir, 'history_regrets.npy'))
    true = true_labels(num_players, num_colluders)
    num_rounds = rewards.shape[1]

    f1_list, plot_times = [], []

    for t in range(1, num_rounds):
        X_t = extract_features_up_to_t(rewards, regrets, t)
        X_scaled = scaler.transform(X_t)
        preds = model.predict(X_scaled)

        f1 = f1_score(true, preds)
        f1_list.append(f1)
        plot_times.append(t)

    return plot_times, f1_list

def rolling_average(values, window=5):
    padded = np.pad(values, (window-1, 0), mode='edge')
    return np.convolve(padded, np.ones(window)/window, mode='valid')


# === Run for all 3 models
results = {}
for model_name, sim_fn in {
    "risky": simulate_risky_collusion_game,
    "normal": simulate_game,
    "auction": simulate_budgeted_auction_game
}.items():
    times, f1 = run_live_f1(model_name, sim_fn)
    results[model_name] = {"times": times, "f1" : f1}

# === Plot Rolling F1 Score
# === Plot Smoothed F1 Score (Gaussian)
plt.figure(figsize=(12, 6))
for model_name, data in results.items():
    # Apply Gaussian filter to smooth the F1 score curve
    smoothed = gaussian_filter1d(data["f1"], sigma=2)  # You can adjust sigma (e.g., 2–4) for more or less smoothing
    plt.plot(data["times"], smoothed, label=f"{model_name} - Smoothed F1", linewidth=2)

plt.xlabel("Round")
plt.ylabel("Smoothed F1 Score")
plt.title("Smoothed F1 Score Over Time: Collusion Detection")
plt.ylim(0, 1.05)
plt.grid(True)
plt.legend()
plt.tight_layout()
#plt.show()


def run_f1_across_trials_xgb(model_name, simulate_fn, n_trials=10, num_players=12, num_colluders=3, max_rounds=300):
    model = joblib.load(f"{model_name}_xgb_model.pkl")
    scaler = joblib.load(f"{model_name}_xgb_scaler.pkl")

    f1_matrix = []

    for trial in range(n_trials):
        save_dir = f"live_xgb_{model_name}_trial_{trial}"
        os.makedirs(save_dir, exist_ok=True)
        simulate_fn(save_dir=save_dir, num_players=num_players, num_colluders=num_colluders)

        rewards = np.load(os.path.join(save_dir, 'history_rewards.npy'))
        regrets = np.load(os.path.join(save_dir, 'history_regrets.npy'))
        true = true_labels(num_players, num_colluders)

        f1_list = []
        for t in range(1, max_rounds):
            X_t = extract_features_up_to_t(rewards, regrets, t)
            X_scaled = scaler.transform(X_t)
            preds = model.predict(X_scaled)
            f1 = f1_score(true, preds)
            f1_list.append(f1)

        f1_matrix.append(f1_list)

    return np.arange(1, max_rounds), np.array(f1_matrix)


# --- Run for each XGBoost model
n_trials = 20
results = {}
for model_name, sim_fn in {
    "risky": simulate_risky_collusion_game,
    "normal": simulate_game,
    "auction": simulate_budgeted_auction_game
}.items():
    times, f1_runs = run_f1_across_trials_xgb(model_name, sim_fn, n_trials=n_trials)
    results[model_name] = {"times": times, "f1_runs": f1_runs}

# --- Plot averaged F1 with std band
plt.figure(figsize=(12, 6))
labels = ['heterogeneous', 'homogeneous', 'auction']
for label, model_name in zip(labels, results.keys()):
    times = results[model_name]["times"]
    f1_runs = results[model_name]["f1_runs"]
    f1_mean = f1_runs.mean(axis=0)
    f1_std = f1_runs.std(axis=0)

    plt.plot(times, f1_mean, label=f"{label} (mean)", linewidth=2)
    plt.fill_between(times, f1_mean - f1_std / 2, f1_mean + f1_std / 2, alpha=0.3)

plt.xlabel("Round")
plt.ylabel("F1 Score")
plt.ylim(0, 1.05)
plt.xlim(0, 300)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
