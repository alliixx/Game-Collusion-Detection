import numpy as np
import os
import joblib
from collections import Counter
from risky_collusion_game import simulate_risky_collusion_game
from game import simulate_game
from auction_game import simulate_budgeted_auction_game
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

def extract_features(save_dir):
    rewards = np.load(os.path.join(save_dir, 'history_rewards.npy'))
    regrets = np.load(os.path.join(save_dir, 'history_regrets.npy'))

    avg_rewards = rewards.mean(axis=1)
    avg_regrets = regrets.mean(axis=1)
    std_rewards = rewards.std(axis=1)
    std_regrets = regrets.std(axis=1)
    regret_over_reward = avg_regrets / (avg_rewards + 1e-6)

    features = np.vstack([
        avg_rewards,
        avg_regrets,
        std_rewards,
        std_regrets,
        regret_over_reward
    ]).T
    return features

def true_labels(num_players, num_colluders):
    return np.array([1 if i < num_colluders else 0 for i in range(num_players)])

def run_experiment(simulate_fn, model_name: str, num_simulations=50, train_size=35):
    X_train, y_train = [], []
    X_test, y_test = [], []

    for i in range(num_simulations):
        num_players = np.random.randint(6, 15)
        num_colluders = np.random.randint(0, max(2, num_players // 2))
        sim_dir = f"{model_name}_data_{i}"

        simulate_fn(
            save_dir=sim_dir,
            seed=i,
            num_players=num_players,
            num_colluders=num_colluders
        )

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

    print(f"\n=== {model_name.upper()} ===")
    print("Training label distribution:", Counter(y_train))

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = xgb.XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss',
        n_estimators=20,
        max_depth=3
    )
    model.fit(X_train_scaled, y_train)

    baseline = DummyClassifier(strategy='most_frequent')
    baseline.fit(X_train_scaled, y_train)
    print("Baseline accuracy:", baseline.score(X_test_scaled, y_test))

    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    conf = confusion_matrix(y_test, y_pred)

    print("Model accuracy:", round(acc, 3))
    print("Confusion matrix:\n", conf)

    model_file = f'{model_name}_xgb_model.pkl'
    scaler_file = f'{model_name}_xgb_scaler.pkl'
    joblib.dump(model, model_file)
    joblib.dump(scaler, scaler_file)
    print(f"Model and scaler saved as {model_file}, {scaler_file}")

    model_size_mb = os.path.getsize(model_file) / (1024 * 1024)
    print(f"Model size: {model_size_mb:.2f} MB, Trees: {len(model.get_booster().get_dump())}, Max depth: {model.get_xgb_params().get('max_depth')}\n")

# === Run on all 3 game variants ===
run_experiment(simulate_risky_collusion_game, model_name='risky')
run_experiment(simulate_game, model_name='normal')
run_experiment(simulate_budgeted_auction_game, model_name='auction')
