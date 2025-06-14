import numpy as np
import matplotlib.pyplot as plt
import random

K = 40
num_players = 100
num_sims = 5000
# m: draw parameter
m = 2
# n_games: no. of games every simulation
n_games = 20

true_elos = np.random.normal(1500, 300, num_players)
true_ranking = [[i, true_elos[i]] for i in range(num_players)]
true_ranking.sort(key=lambda x: x[1], reverse=True)
# player_id -> (rank, elo)
dic_true = {true_ranking[i][0]: (i + 1, true_ranking[i][1]) for i in range(num_players)}


def group_matching(ranking: list, bin_size=200):
    groups = {}  # {bin_id: [player1, player2, ...]}

    # Step 1: Assign players to bins
    for player_id, elo in ranking:
        bin_id = int(elo) // bin_size  # e.g. 1500 // 200 = 7
        if bin_id not in groups:
            groups[bin_id] = []
        groups[bin_id].append(player_id)

    # Step 2: Match within each group
    pairs = []
    for group in groups.values():
        random.shuffle(group)
        for i in range(0, len(group) - 1, 2):
            pairs.append((group[i], group[i + 1]))

    return pairs


def metric(dic_true, dic_est, rank=True):
    if rank:
        difference = [abs(dic_true[i][0] - dic_est[i][0]) for i in range(num_players)]
    else:
        difference = [abs(dic_true[i][1] - dic_est[i][1]) for i in range(num_players)]
    return sum(difference) / num_players


res = []
elos = [1500 for i in range(num_players)]
ranking = [[i, elos[i]] for i in range(num_players)]

for _ in range(num_sims):
    # Divide players into groups based on Elo ranges (e.g., bins of 200 points),
    # then match players within each group using simple methods (random)
    match_lis = group_matching(ranking)
    for i, j in match_lis:
        # sigma = 200, z = r_i - r_j
        g_i = (10**(0.5*(elos[i] - elos[j])/200) + 1/2 * m) / (10**(0.5*(elos[i] - elos[j])/200) + 10**(-0.5*(elos[i] - elos[j])/200) + m)
        g_j = (10**(0.5*(elos[j] - elos[i])/200) + 1/2 * m) / (10**(0.5*(elos[j] - elos[i])/200) + 10**(-0.5*(elos[j] - elos[i])/200) + m)
        
        p_H = 10**(0.5*(true_elos[i] - true_elos[j])/200) / (
            10**(0.5*(true_elos[i] - true_elos[j])/200) + 10**(-0.5*(true_elos[i] - true_elos[j])/200) + m)      # home wins
        p_A = 10**(-0.5*(true_elos[i] - true_elos[j])/200) / (
            10**(0.5*(true_elos[i] - true_elos[j])/200) + 10**(-0.5*(true_elos[i] - true_elos[j])/200) + m)      # away wins
        p_D = 1 - p_H - p_A                                                                                      # draw

        s_i = np.random.choice([1, 0, 0.5], p=[p_H, p_A, p_D])
        s_j = 1 - s_i

        elos[i] += K * (s_i - g_i)
        elos[j] += K * (s_j - g_j)
    ranking = [[i, elos[i]] for i in range(num_players)]
    ranking.sort(key=lambda x: x[1], reverse=True)
    dic_est = {ranking[i][0]: (i + 1, ranking[i][1]) for i in range(num_players)}
    res.append(metric(dic_true, dic_est, rank=True))
    # changed because perason coefficients could give wrong impression
    # like we could have the order of people wrong and it would be perfect
    # as long as the set of elos is equal

    # rank metric converges but wont converge further because we dont have draws
    # for people with similar elo, their prob of winning would be close to 0.5
    # and I suppose they would draw a lot, but they end up winning or losing
    # causing them to oscillate rankings

    # elo metric initially converges but actually diverge perhaps because in 
    # our population, we have players who are much better than the others
    # and end up getting really high and low elos

print(dic_true)
print("Estimate:")
print(dic_est)

plt.plot(range(len(res)), res)
plt.title(f"Elo Metric, Group, {num_sims} Rounds")
plt.xlabel("Iterations")
plt.ylabel("Metric")
# plt.ylim(0, 300)
plt.grid(True)
plt.tight_layout()
plt.show()
