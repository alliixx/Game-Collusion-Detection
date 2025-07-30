import numpy as np
import os
import matplotlib.pyplot as plt

def compute_dynamic_collusion_payoffs(actions, colluders, honest, influence_weights, num_actions,
                                      collusion_reward, collusion_penalty=0.2, collusion_success=True):
    num_players = len(actions)
    payoffs = np.zeros(num_players)

    if len(colluders) > 0:
        target_action = np.bincount(actions[colluders]).argmax()
        for idx in colluders:
            if actions[idx] == target_action:
                if collusion_success:
                    payoffs[idx] = collusion_reward * influence_weights[idx] + 0.1 * np.random.rand()
                else:
                    payoffs[idx] = 0.3 + 0.1 * np.random.rand()
            else:
                payoffs[idx] = collusion_penalty * np.random.rand()

    for i in honest:
        base = 0.4 + 0.5 * (actions[i] == i % num_actions)
        bonus = 0.2 * (len(colluders) > 0 and not collusion_success)
        payoffs[i] = base + bonus + 0.1 * np.random.rand()

    return payoffs

def simulate_dynamic_reward_game(
    save_dir='game_data_dynamic_reward',
    num_players=10,
    num_colluders=3,
    num_actions=4,
    num_rounds=1000,
    eta=0.1,
    base_reward=1.0,
    collusion_penalty=0.2,
    collusion_success_prob=0.7,
    seed=None
):
    if seed is not None:
        np.random.seed(seed)

    players = np.arange(num_players)
    colluders = players[:num_colluders]
    honest = players[num_colluders:]

    influence_weights = np.ones(num_players)
    if len(colluders) > 0:
        influence_weights[colluders] = np.random.uniform(1.05, 1.25, size=len(colluders))
        collusion_adherence = np.random.uniform(0.6, 0.95, size=len(colluders))

    shared_weights = np.ones(num_actions)
    weights = np.ones((num_players, num_actions))

    history_rewards = np.zeros((num_players, num_rounds))
    history_regrets = np.zeros((num_players, num_rounds))
    history_actions = np.zeros((num_players, num_rounds), dtype=int)
    history_dynamic_rewards = np.zeros(num_rounds)

    for t in range(num_rounds):
        colluder_probs = shared_weights / shared_weights.sum()
        probs = weights / weights.sum(axis=1, keepdims=True)

        collusion_reward = base_reward + 0.5 * np.sin(t / 50) + 0.2 * np.random.randn()
        collusion_reward = max(0.1, collusion_reward)
        history_dynamic_rewards[t] = collusion_reward

        if len(colluders) > 0:
            collusion_target_action = np.random.choice(num_actions, p=colluder_probs)
            collusion_success = np.random.rand() < collusion_success_prob
        else:
            collusion_target_action = None
            collusion_success = False

        actions = np.zeros(num_players, dtype=int)
        for i in range(num_players):
            if i in colluders:
                idx = np.where(colluders == i)[0][0]
                if np.random.rand() < collusion_adherence[idx]:
                    actions[i] = collusion_target_action
                else:
                    actions[i] = np.random.choice(num_actions, p=colluder_probs)
            else:
                actions[i] = np.random.choice(num_actions, p=probs[i])

        history_actions[:, t] = actions

        rewards = compute_dynamic_collusion_payoffs(
            actions, colluders, honest, influence_weights, num_actions,
            collusion_reward, collusion_penalty, collusion_success
        )

        for i in range(num_players):
            counterfactual_payoffs = np.zeros(num_actions)
            for a in range(num_actions):
                counterfactual = actions.copy()
                counterfactual[i] = a
                counterfactual_payoffs[a] = compute_dynamic_collusion_payoffs(
                    counterfactual, colluders, honest, influence_weights, num_actions,
                    collusion_reward, collusion_penalty, collusion_success
                )[i]

            best = np.max(counterfactual_payoffs)
            regret = best - rewards[i]

            history_rewards[i, t] = rewards[i]
            history_regrets[i, t] = regret

            if i in colluders:
                shared_weights *= np.exp(eta * counterfactual_payoffs)
            else:
                weights[i] *= np.exp(eta * counterfactual_payoffs)

    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, 'history_rewards.npy'), history_rewards)
    np.save(os.path.join(save_dir, 'history_regrets.npy'), history_regrets)
    np.save(os.path.join(save_dir, 'history_actions.npy'), history_actions)
    np.save(os.path.join(save_dir, 'history_dynamic_rewards.npy'), history_dynamic_rewards)

    print(f"Saved to {save_dir}")

    # === Plot 1: Dynamic collusion reward over time ===
    # plt.figure(figsize=(10, 5))
    # plt.plot(history_dynamic_rewards)
    # plt.title("Dynamic Collusion Reward Over Time")
    # plt.xlabel("Round")
    # plt.ylabel("Collusion Reward")
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()

    # # === Plot 2: Cumulative reward per player ===
    # cumulative_rewards = history_rewards.sum(axis=1)
    # plt.figure(figsize=(10, 6))
    # for i in range(num_players):
    #     plt.bar(i, cumulative_rewards[i], color='red' if i < num_colluders else 'blue')
    # plt.title("Cumulative Reward per Player")
    # plt.xlabel("Player ID")
    # plt.ylabel("Total Reward")
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()

    # # === Plot 3: Average regret per player ===
    # average_regrets = history_regrets.mean(axis=1)
    # plt.figure(figsize=(10, 6))
    # for i in range(num_players):
    #     plt.bar(i, average_regrets[i], color='red' if i < num_colluders else 'blue')
    # plt.title("Average Regret per Player")
    # plt.xlabel("Player ID")
    # plt.ylabel("Average Regret")
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()

    # # === Plot 4: Regret over time per player ===
    # plt.figure(figsize=(10, 5))
    # for i in range(num_players):
    #     avg_regret = np.cumsum(history_regrets[i]) / (np.arange(1, num_rounds + 1))
    #     plt.plot(avg_regret, label=f'Player {i}', color='red' if i < num_colluders else 'blue')
    # plt.title("Average Regret Over Time")
    # plt.xlabel("Round")
    # plt.ylabel("Average Regret")
    # plt.axhline(0, color='black', linestyle='--')
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()

    return save_dir

# === Sample usage ===
if __name__ == "__main__":
    simulate_dynamic_reward_game(seed=42)
