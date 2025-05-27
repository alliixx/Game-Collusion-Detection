import numpy as np
import os
import matplotlib.pyplot as plt

def compute_dynamic_collusion_payoffs(actions, colluders, honest, influence_weights, num_actions,
                                      collusion_reward=1.0, collusion_penalty=0.2, collusion_success=True,
                                      collusion_weights=None):
    num_players = len(actions)
    payoffs = np.zeros(num_players)

    if len(colluders) > 0:
        target_action = np.bincount(actions[colluders]).argmax()
        for idx in colluders:
            if actions[idx] == target_action:
                if collusion_success:
                    adherence_strength = collusion_weights[idx] if collusion_weights is not None else 1.0
                    noise = np.random.normal(0, 0.3)
                    payoffs[idx] = max(0, collusion_reward * adherence_strength + noise)
                else:
                    payoffs[idx] = 0.3 * np.random.rand()
            else:
                payoffs[idx] = collusion_penalty * np.random.rand()

    for i in honest:
        aligned = actions[i] == (i % num_actions)
        base = 0.4 + 0.5 * aligned
        noise_factor = 0.2 * (len(colluders) > 0 and not collusion_success)
        payoffs[i] = base + noise_factor + np.random.normal(0, 0.2)

    return payoffs

def simulate_risky_collusion_game(
    save_dir='game_data_soft_risky',
    num_players=10,
    num_colluders=3,
    num_actions=4,
    num_rounds=1000,
    eta=0.1,
    collusion_reward=1.0,
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
    collusion_weights = np.ones(num_players)
    collusion_adherence = np.zeros(num_players)

    if len(colluders) > 0:
        influence_weights[colluders] = np.random.uniform(1.05, 1.25, size=len(colluders))
        collusion_weights[colluders] = np.random.uniform(0.6, 1.2, size=len(colluders))
        collusion_adherence[colluders] = np.random.uniform(0.5, 0.8, size=len(colluders))

    shared_weights = np.ones(num_actions)
    weights = np.ones((num_players, num_actions))

    history_rewards = np.zeros((num_players, num_rounds))
    history_regrets = np.zeros((num_players, num_rounds))
    history_actions = np.zeros((num_players, num_rounds), dtype=int)

    for t in range(num_rounds):
        colluder_probs = shared_weights / shared_weights.sum()
        probs = weights / weights.sum(axis=1, keepdims=True)

        collusion_target_action = np.random.choice(num_actions, p=colluder_probs) if len(colluders) > 0 else np.random.choice(num_actions)
        collusion_success = np.random.rand() < collusion_success_prob

        actions = np.zeros(num_players, dtype=int)
        for i in range(num_players):
            if i in colluders:
                if np.random.rand() < collusion_adherence[i]:
                    actions[i] = collusion_target_action
                else:
                    if np.random.rand() < 0.25:
                        actions[i] = np.random.randint(num_actions)
                    else:
                        actions[i] = np.random.choice(num_actions, p=colluder_probs)
            else:
                if np.random.rand() < 0.3:
                    actions[i] = collusion_target_action
                elif np.random.rand() < 0.25:
                    actions[i] = np.random.randint(num_actions)
                else:
                    actions[i] = np.random.choice(num_actions, p=probs[i])

        history_actions[:, t] = actions

        rewards = compute_dynamic_collusion_payoffs(
            actions, colluders, honest, influence_weights, num_actions,
            collusion_reward, collusion_penalty, collusion_success,
            collusion_weights=collusion_weights
        )

        rewards += np.random.normal(0, 0.5, size=rewards.shape)

        regrets = np.zeros(num_players)
        for i in range(num_players):
            counterfactual_payoffs = np.zeros(num_actions)
            for a in range(num_actions):
                counterfactual = actions.copy()
                counterfactual[i] = a
                counterfactual_payoffs[a] = compute_dynamic_collusion_payoffs(
                    counterfactual, colluders, honest, influence_weights, num_actions,
                    collusion_reward, collusion_penalty, collusion_success,
                    collusion_weights=collusion_weights
                )[i]

            best = np.max(counterfactual_payoffs)
            regrets[i] = best - rewards[i]

            if i in colluders:
                shared_weights *= np.exp(eta * counterfactual_payoffs)
            else:
                weights[i] *= np.exp(eta * counterfactual_payoffs)

        history_rewards[:, t] = rewards
        history_regrets[:, t] = regrets

    # os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, 'history_rewards.npy'), history_rewards)
    np.save(os.path.join(save_dir, 'history_regrets.npy'), history_regrets)
    np.save(os.path.join(save_dir, 'history_actions.npy'), history_actions)

    # print(f"Saved to {save_dir}")

    cumulative_rewards = history_rewards.sum(axis=1)
    average_regrets = history_regrets.mean(axis=1)

    # plt.figure(figsize=(10, 6))
    # for i in range(num_players):
    #     color = 'red' if i in colluders else 'blue'
    #     plt.bar(i, cumulative_rewards[i], color=color)
    # plt.title("Cumulative Reward per Player")
    # plt.xlabel("Player ID")
    # plt.ylabel("Total Reward")
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()

    # plt.figure(figsize=(10, 6))
    # for i in range(num_players):
    #     color = 'red' if i in colluders else 'blue'
    #     plt.bar(i, average_regrets[i], color=color)
    # plt.title("Average Regret per Player")
    # plt.xlabel("Player ID")
    # plt.ylabel("Average Regret")
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()

    # plt.figure(figsize=(10, 5))
    # for i in range(num_players):
    #     color = 'red' if i in colluders else 'blue'
    #     plt.plot(np.cumsum(history_rewards[i]), label=f'Player {i}', color=color)
    # plt.title("Cumulative Reward Over Time")
    # plt.xlabel("Round")
    # plt.ylabel("Cumulative Reward")
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()

    # plt.figure(figsize=(10, 5))
    # for i in range(num_players):
    #     color = 'red' if i in colluders else 'blue'
    #     avg_regret = np.cumsum(history_regrets[i]) / (np.arange(1, num_rounds + 1))
    #     plt.plot(avg_regret, label=f'Player {i}', color=color)
    # plt.title("Average Regret Per Round")
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
    simulate_risky_collusion_game(seed=42)
