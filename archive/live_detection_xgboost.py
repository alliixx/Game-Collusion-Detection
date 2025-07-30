import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
from games.risky_collusion_game import simulate_risky_collusion_game
from games.game import simulate_game
from games.auction_game import simulate_budgeted_auction_game

def true_labels(num_players, num_colluders):
    return np.array([1 if i < num_colluders else 0 for i in range(num_players)])

def live_predict(model, scaler, rewards, regrets, threshold=0.95, warmup_rounds=5, sustain_rounds=3):
    num_players, num_rounds = rewards.shape
    final_preds = np.full(num_players, -1)
    final_conf = np.zeros(num_players)
    prediction_time = np.zeros(num_players)
    sustained_count = np.zeros(num_players, dtype=int)

    for t in range(1, num_rounds):
        avg_rewards = rewards[:, :t+1].mean(axis=1)
        avg_regrets = regrets[:, :t+1].mean(axis=1)
        std_rewards = rewards[:, :t+1].std(axis=1)
        std_regrets = regrets[:, :t+1].std(axis=1)
        regret_over_reward = avg_regrets / (avg_rewards + 1e-6)

        X_t = np.vstack([
            avg_rewards,
            avg_regrets,
            std_rewards,
            std_regrets,
            regret_over_reward
        ]).T
        X_scaled = scaler.transform(X_t)

        probs = model.predict_proba(X_scaled)[:, 1]
        print(f"\nRound {t}: probs =", np.round(probs, 2))

        if t < warmup_rounds:
            print(f"  Skipping predictions (warmup round {t})")
            continue

        for i in range(num_players):
            if final_preds[i] != -1:
                continue

            prob = probs[i]
            if prob > threshold or prob < 1 - threshold:
                sustained_count[i] += 1
            else:
                sustained_count[i] = 0

            print(f"  Player {i} | Prob: {prob:.2f} | Sustained: {sustained_count[i]}/{sustain_rounds}")

            if sustained_count[i] >= sustain_rounds:
                final_preds[i] = int(prob > 0.5)
                final_conf[i] = prob
                prediction_time[i] = t
                print(f"  → Final prediction made for Player {i}: {final_preds[i]} at round {t} with conf {prob:.2f}")

        if np.all(final_preds != -1):
            print("All predictions made — stopping early.")
            break

    return final_preds, final_conf, prediction_time

# === Load model and scaler ===
model = joblib.load('normal_xgb_model.pkl')
scaler = joblib.load('normal_xgb_scaler.pkl')

# === Run live game ===
save_dir = "live_game"
num_players = 12
num_colluders = 3
simulate_game(save_dir=save_dir, seed=42, num_players=num_players, num_colluders=num_colluders)

rewards = np.load(os.path.join(save_dir, 'history_rewards.npy'))
regrets = np.load(os.path.join(save_dir, 'history_regrets.npy'))
true = true_labels(num_players, num_colluders)

# === Predict ===
preds, confs, times = live_predict(model, scaler, rewards, regrets)

# === Output ===
print("\n=== Final Predictions ===")
print("Predicted colluders:", np.where(preds == 1)[0])
print("True colluders:     ", np.where(true == 1)[0])
print("Prediction times (rounds):", times.astype(int))

# === Plot ===
plt.figure(figsize=(10, 6))
plt.bar(range(num_players), confs, color=["red" if p == 1 else "blue" for p in preds])
plt.axhline(0.95, linestyle='--', color='gray', label="Confidence Threshold")
plt.title("Final Prediction Confidence per Player")
plt.xlabel("Player ID")
plt.ylabel("Predicted Collusion Confidence")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
