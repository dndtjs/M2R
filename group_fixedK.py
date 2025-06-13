import numpy as np
import random
import matplotlib.pyplot as plt
from collections import defaultdict

K = 40
num_players = 100
num_sims = 1000

true_elos = np.random.normal(1500, 300, num_players)
true_ranking = [[i, true_elos[i]] for i in range(num_players)]
true_ranking.sort(key=lambda x: x[1], reverse=True)
# player_id -> (rank, elo)
dic_true = {true_ranking[i][0]: (i + 1, true_ranking[i][1]) for i in range(num_players)}
dic_count = {i: 0 for i in range(num_players)}
dic_K = {i: 40 for i in range(num_players)}

def metric(dic_true, dic_est, rank=True):
    if rank:
        difference = [abs(dic_true[i][0] - dic_est[i][0]) for i in range(num_players)]
    else:
        difference = [abs(dic_true[i][1] - dic_est[i][1]) for i in range(num_players)]
    return sum(difference) / len(difference)

res = []
elos = [1500 for i in range(num_players)]
ranking = [[i, elos[i]] for i in range(num_players)]

for n in range(num_sims):
    # rank the current elo and match no.1 with no.2, no.3 with no.4, ...
    clusters = defaultdict(list)
    for i, elo in enumerate(elos):
        key = int(elo // 200)
        clusters[key].append(i)
    matches = []
    for players in clusters.values():
        if len(players) < 2:
            continue
        random.shuffle(players)
        for i in range(0, len(players) - 1, 2):
            matches.append((players[i], players[i + 1]))
    for i, j in matches:
        e_i = 1 / (1 + 10 ** ((elos[j] - elos[i]) / 400))
        e_j = 1 - e_i
        # assume no drawing
        s_i = np.random.binomial(n=1, p=1 / (1 + 10 ** ((true_elos[j] - true_elos[i]) / 400)))
        s_j = 1 - s_i
        elos[i] += dic_K[i] * (s_i - e_i)
        elos[j] += dic_K[j] * (s_j - e_j)
        dic_count[i] += 1
        dic_count[j] += 1
    ranking = [[i, elos[i]] for i in range(num_players)]
    ranking.sort(key=lambda x: x[1], reverse=True)
    dic_est = {ranking[i][0]: (i + 1, ranking[i][1]) for i in range(num_players)}
    res.append(metric(dic_true, dic_est, rank=False))
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
plt.title(f"Elo Metric, Group, {num_sims} Rounds, No Draw")
plt.xlabel("Number of Rounds")
plt.ylabel("Mean Elo Error")
plt.grid(True)
plt.tight_layout()
plt.show()