import numpy as np
import os
import matplotlib.pyplot as plt

def simulate_budgeted_auction_game(
    save_dir='budgeted_auction_game',
    num_players=10,
    num_colluders=3,
    num_items=300,
    base_item_value=1.0,
    initial_budget=10.0,
    honest_participation_prob=0.7,
    seed=None
):
    # if seed is not None:
    #     np.random.seed(seed)

    players = np.arange(num_players)
    colluders = players[:num_colluders] if num_colluders > 0 else np.array([], dtype=int)
    honest = players[num_colluders:]
    budgets = np.full(num_players, initial_budget)

    profits = np.zeros((num_players, num_items))
    bids_history = np.zeros((num_players, num_items))
    regrets = np.zeros((num_players, num_items))


    if num_colluders > 0:
        colluder_allocation = [colluders[i % num_colluders] for i in range(num_items)]
    else:
        colluder_allocation = [None for _ in range(num_items)]

    for t in range(num_items):
        item_value = base_item_value + 0.1 * np.random.randn()
        item_value = max(0.1, item_value)

        bids = np.zeros(num_players)
        values = np.zeros(num_players)

        for i in range(num_players):
            if budgets[i] <= 1e-3:
                continue

            if i in colluders:
                participation_prob = min(0.2 + 0.01 * t, 1.0)
                if np.random.rand() > participation_prob:
                    continue

                if i == colluder_allocation[t]:
                    value = item_value * (0.9 + 0.05 * np.random.rand())
                    bid = min(value, budgets[i])
                else:
                    value = item_value * np.random.uniform(0.1, 0.3)
                    bid = min(value, budgets[i])
            else:
                if np.random.rand() > honest_participation_prob:
                    continue
                strategy = np.random.choice(['aggressive', 'conservative', 'random'], p=[0.4, 0.4, 0.2])
                if strategy == 'aggressive':
                    value = item_value * (0.9 + 0.1 * np.random.rand())
                elif strategy == 'conservative':
                    value = item_value * (0.6 + 0.1 * np.random.rand())
                else:
                    value = item_value * np.random.uniform(0.5, 1.0)
                bid = min(value, budgets[i])

            bids[i] = bid
            values[i] = value

        BID_THRESHOLD = 1e-3
        positive_bidders = np.array([i for i in range(num_players) if bids[i] > BID_THRESHOLD])

        if len(positive_bidders) < 2:
            #print(f"Round {t}: No valid positive bids — skipping auction.")
            continue


        sorted_indices = positive_bidders[np.argsort(-bids[positive_bidders])]


        winner = sorted_indices[0]
        second_price = bids[sorted_indices[1]] if len(sorted_indices) > 1 else 0.0

        payment = second_price
        value_winner = values[winner]

        if budgets[winner] >= payment:
            profit = value_winner - payment
            profits[winner, t] = profit
            budgets[winner] -= payment
            #print(f"Round {t}: Player {winner} wins at price {payment:.2f}, remaining budget: {budgets[winner]:.2f}")
        else:
            profit = 0

        bids_history[:, t] = bids

        for i in range(num_players):
            value = values[i]
            budget = budgets[i] + (payment if i == winner else 0)
            actual_profit = profits[i, t]

            if i == winner:
                # Could they have bid just above the 3rd highest bidder instead?
                best_lower_bid = second_price
                for alt in sorted_indices[2:]:
                    potential_payment = bids[alt]
                    if potential_payment < best_lower_bid:
                        best_lower_bid = potential_payment
                alt_payment = best_lower_bid
                alt_profit = value - alt_payment if alt_payment <= budget else actual_profit
                regret = max(alt_profit - actual_profit, 0)
            else:
                # Could they have won by bidding just above winner?
                min_required = bids[winner] + 0.01
                if min_required <= budget:
                    alt_profit = value - bids[winner]
                    regret = max(alt_profit - 0, 0)
                else:
                    regret = 0

            regrets[i, t] = regret

    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, "history_rewards.npy"), profits)
    np.save(os.path.join(save_dir, "history_bids.npy"), bids_history)
    np.save(os.path.join(save_dir, "history_regrets.npy"), regrets)

    # === Plotting ===
    cumulative_rewards = profits.sum(axis=1)
    cum_profits = profits.cumsum(axis=1)
    cumulative_regrets = regrets.sum(axis=1)

    # plt.figure(figsize=(10, 5))
    # for i in range(num_players):
    #     color = 'red' if i in colluders else 'blue'
    #     plt.bar(i, cumulative_rewards[i], color=color)
    # plt.xlabel("Player ID")
    # plt.ylabel("Cumulative Reward")
    # plt.title("Cumulative Reward per Player")
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()

    # plt.figure(figsize=(10, 5))
    # for i in range(num_players):
    #     color = 'red' if i in colluders else 'blue'
    #     plt.bar(i, cumulative_regrets[i], color=color)
    # plt.xlabel("Player ID")
    # plt.ylabel("Cumulative Regret")
    # plt.title("Cumulative Regret per Player")
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()

    # plt.figure(figsize=(10, 6))
    # for i in range(num_players):
    #     color = 'red' if i in colluders else 'blue'
    #     plt.plot(cum_profits[i], label=f'Player {i}', color=color)
    # plt.title("Cumulative Profit Over Time")
    # plt.xlabel("Item Round")
    # plt.ylabel("Cumulative Profit")
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()

    # plt.figure(figsize=(10, 6))
    # for i in range(num_players):
    #     color = 'red' if i in colluders else 'blue'
    #     avg_regret = np.cumsum(regrets[i]) / (np.arange(1, num_items + 1))
    #     plt.plot(avg_regret, label=f'Player {i}', color=color)
    # plt.title("Average Regret Over Time")
    # plt.xlabel("Item Round")
    # plt.ylabel("Average Regret")
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()




    return save_dir

# === Sample usage ===
if __name__ == "__main__":
    simulate_budgeted_auction_game()
