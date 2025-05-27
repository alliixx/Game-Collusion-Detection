# import numpy as np
# import os
# import joblib
# import matplotlib.pyplot as plt
# from game import simulate_game
# from risky_collusion_game import simulate_risky_collusion_game
# from auction_game import simulate_budgeted_auction_game

# def true_labels(num_players, num_colluders):
#     return np.array([1 if i < num_colluders else 0 for i in range(num_players)])

# def extract_features_up_to_t(rewards, regrets, t):
#     avg_rewards = rewards[:, :t+1].mean(axis=1)
#     avg_regrets = regrets[:, :t+1].mean(axis=1)
#     return np.vstack([avg_rewards, avg_regrets]).T

# def periodic_kmeans_predictions(kmeans_model, mapping, rewards, regrets, print_step=50, plot_step=100):
#     num_players, num_rounds = rewards.shape

#     for t in range(print_step, num_rounds + 1, print_step):
#         X_t = extract_features_up_to_t(rewards, regrets, t-1)
#         cluster_labels = kmeans_model.predict(X_t)
#         preds = np.array([mapping[c] for c in cluster_labels])

#         print(f"Round {t}: Predicted colluders: {np.where(preds == 1)[0]}")

#         # Plot every `plot_step` rounds
#         if t % plot_step == 0:
#             avg_rewards = X_t[:, 0]
#             avg_regrets = X_t[:, 1]

#             plt.figure(figsize=(8, 6))
#             colors = ['red' if label == 1 else 'green' for label in preds]
#             plt.scatter(avg_rewards, avg_regrets, c=colors, s=100, edgecolors='black')
#             for i in range(num_players):
#                 plt.annotate(f"{i}", (avg_rewards[i], avg_regrets[i]), textcoords="offset points", xytext=(5, 5))
#             plt.xlabel("Average Reward")
#             plt.ylabel("Average Regret")
#             plt.title(f"KMeans Predictions at Round {t}")
#             plt.grid(True)
#             plt.tight_layout()
#             plt.show()

# # === Load saved model and mapping ===
# kmeans = joblib.load("kmeans_normal_model.pkl")
# mapping = joblib.load("kmeans_normal_mapping.pkl")

# # === Run a new live game ===
# live_dir = "live_kmeans_game"
# num_players = 12
# num_colluders = 3
# simulate_game(save_dir=live_dir, num_players=num_players, num_colluders=num_colluders)

# # === Load game data ===
# rewards = np.load(os.path.join(live_dir, 'history_rewards.npy'))
# regrets = np.load(os.path.join(live_dir, 'history_regrets.npy'))
# true = true_labels(num_players, num_colluders)

# # === Run periodic predictions ===
# periodic_kmeans_predictions(kmeans, mapping, rewards, regrets, print_step=10, plot_step=10)



import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
from game import simulate_game
from auction_game import simulate_budgeted_auction_game
from risky_collusion_game import simulate_risky_collusion_game
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import f1_score

def true_labels(num_players, num_colluders):
    return np.array([1 if i < num_colluders else 0 for i in range(num_players)])

def extract_features_up_to_t(rewards, regrets, t):
    avg_rewards = rewards[:, :t+1].mean(axis=1)
    avg_regrets = regrets[:, :t+1].mean(axis=1)
    return np.vstack([avg_rewards, avg_regrets]).T

def run_live_f1_kmeans(model_name, simulate_fn):
    print(f"\n=== Evaluating {model_name.upper()} KMeans Model ===")

    # Load model + mapping
    kmeans = joblib.load(f'kmeans_{model_name}_model.pkl')
    mapping = joblib.load(f'kmeans_{model_name}_mapping.pkl')

    # Simulate game
    save_dir = f"live_kmeans_{model_name}_game"
    num_players = 12
    num_colluders = 3
    simulate_fn(save_dir=save_dir, num_players=num_players, num_colluders=num_colluders)

    # Load game data
    rewards = np.load(os.path.join(save_dir, 'history_rewards.npy'))
    regrets = np.load(os.path.join(save_dir, 'history_regrets.npy'))
    true = true_labels(num_players, num_colluders)
    num_rounds = rewards.shape[1]

    f1_list, plot_times = [], []

    for t in range(1, num_rounds):
        X_t = extract_features_up_to_t(rewards, regrets, t)
        clusters = kmeans.predict(X_t)
        preds = np.array([mapping[c] for c in clusters])
        if t == num_rounds-1:
            print(f'true: {true}, pred: {preds}')

        f1 = f1_score(true, preds)
        f1_list.append(f1)
        plot_times.append(t)

    return plot_times, f1_list

# === Run KMeans F1 for each game
results = {}
for model_name, sim_fn in {
    "risky": simulate_risky_collusion_game,
    "normal": simulate_game,
    "auction": simulate_budgeted_auction_game
}.items():
    times, f1 = run_live_f1_kmeans(model_name, sim_fn)
    results[model_name] = {"times": times, "f1": f1}

# === Plot Gaussian-Smoothed F1 Score Over Time
plt.figure(figsize=(12, 6))
model = ['heterogenous', 'homogeneous', 'auction']
for modelname, model_name in zip(model, results.keys()):
    smoothed = gaussian_filter1d(results[model_name]["f1"], sigma=2)
    plt.plot(results[model_name]["times"], smoothed, label=f"{modelname}", linewidth=2)

plt.xlabel("Round")
plt.ylabel("F1 Score")
plt.title("F1 Score Over Time: KMeans Collusion Detection")
plt.ylim(0, 1.05)
plt.xlim(0, 600)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
