import numpy as np
import os
import matplotlib.pyplot as plt

def compute_payoffs(actions, colluders, honest, num_actions, collusion_reward=1.0, collusion_penalty=0.2):
    num_players = len(actions)
    payoffs = np.zeros(num_players)

    # Colluder logic: benefit if all colluders play same action
    if len(colluders) > 0:
        colluder_actions = actions[colluders]
        if np.all(colluder_actions == colluder_actions[0]):
            reward = collusion_reward + 0.2 * np.random.rand()
            payoffs[colluders] = reward
        else:
            penalty = collusion_penalty * np.random.rand()
            payoffs[colluders] = penalty


    # Honest player logic: noisy function of consistent strategy
    for i in honest:
        payoffs[i] = 0.4 + 0.5 * (actions[i] == i % num_actions) + 0.1 * np.random.rand()

    return payoffs

def simulate_game(
    save_dir='game_data',
    num_players=10,
    num_colluders=3,
    num_actions=4,
    num_rounds=1000,
    eta=0.1,
    collusion_reward=1.0,
    collusion_penalty=0.2,
    seed=None
):
    """
    Simulate a repeated multiplayer game with colluders vs honest players.
    Saves history_rewards.npy, history_regrets.npy, and history_actions.npy to save_dir.
    """
    if seed is not None:
        np.random.seed(seed)

    players = np.arange(num_players)
    colluding_players = players[:num_colluders]
    honest_players = players[num_colluders:]

    shared_colluder_weights = np.ones(num_actions)
    weights = np.ones((num_players, num_actions))

    history_rewards = np.zeros((num_players, num_rounds))
    history_regrets = np.zeros((num_players, num_rounds))
    history_actions = np.zeros((num_players, num_rounds), dtype=int)

    for t in range(num_rounds):
        if len(colluding_players) > 0:
            colluder_probs = shared_colluder_weights / shared_colluder_weights.sum()

        probs = weights / weights.sum(axis=1, keepdims=True)

        actions = np.zeros(num_players, dtype=int)
        for i in range(num_players):
            if i in colluding_players:
                actions[i] = np.random.choice(num_actions, p=colluder_probs)
            else:
                actions[i] = np.random.choice(num_actions, p=probs[i])

        history_actions[:, t] = actions
        rewards = compute_payoffs(actions, colluding_players, honest_players, num_actions, collusion_reward, collusion_penalty)

        for i in range(num_players):
            counterfactual_payoffs = np.zeros(num_actions)
            for a in range(num_actions):
                counterfactual_action = actions.copy()
                counterfactual_action[i] = a
                counterfactual_payoffs[a] = compute_payoffs(counterfactual_action, colluding_players, honest_players, num_actions, collusion_reward, collusion_penalty)[i]

            best_response = np.max(counterfactual_payoffs)
            regret = best_response - rewards[i]

            history_rewards[i, t] = rewards[i]
            history_regrets[i, t] = regret

            if i in colluding_players and len(colluding_players) > 0:
                shared_colluder_weights *= np.exp(eta * counterfactual_payoffs)
            else:
                weights[i] *= np.exp(eta * counterfactual_payoffs)

    # os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, 'history_rewards.npy'), history_rewards)
    np.save(os.path.join(save_dir, 'history_regrets.npy'), history_regrets)
    np.save(os.path.join(save_dir, 'history_actions.npy'), history_actions)

    # print(f"Saved history files to {save_dir}/")

    # ==== PLOTTING ====

    cumulative_rewards = history_rewards.sum(axis=1)
    average_regrets = history_regrets.mean(axis=1)

    # Cumulative Reward Plot
    # plt.figure(figsize=(10, 6))
    # for i in range(num_players):
    #     color = 'red' if i in colluding_players else 'blue'
    #     plt.bar(i, cumulative_rewards[i], color=color)
    # plt.title("Cumulative Reward per Player")
    # plt.xlabel("Player ID")
    # plt.ylabel("Total Reward")
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()

    # # Average Regret Plot
    # plt.figure(figsize=(10, 6))
    # for i in range(num_players):
    #     color = 'red' if i in colluding_players else 'blue'
    #     plt.bar(i, average_regrets[i], color=color)
    # plt.title("Average Regret per Player")
    # plt.xlabel("Player ID")
    # plt.ylabel("Average Regret")
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()

    # plt.figure(figsize=(10, 5))
    # for i in range(num_players):
    #     color = 'red' if i in colluding_players else 'blue'
    #     plt.plot(np.cumsum(history_rewards[i]), label=f'Player {i}', color=color)
    # plt.title("Cumulative Reward Over Time")
    # plt.xlabel("Round")
    # plt.ylabel("Cumulative Reward")
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()

    # # Average Regret Per Round
    # plt.figure(figsize=(10, 5))
    # for i in range(num_players):
    #     color = 'red' if i in colluding_players else 'blue'
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

# Example usage
if __name__ == "__main__":
    simulate_game(save_dir='game_data')


