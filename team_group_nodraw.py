import numpy as np
import matplotlib.pyplot as plt
import random

K = 40
num_players = 100
# number of people in a team
n_teams = 5
num_sims = 100000
sgm= 30  # I tried 30 is the best one for case of no drawings, the graph in overleaf use sgm=400

true_elos = np.random.normal(1500, 300, num_players)
true_ranking = [[i, true_elos[i]] for i in range(num_players)]
true_ranking.sort(key=lambda x: x[1], reverse=True)
# player_id -> (rank, elo)
dic_true = {true_ranking[i][0]: (i + 1, true_ranking[i][1]) for i in range(num_players)}


def metric(dic_true, dic_est, rank=True):
    if rank:
        difference = [abs(dic_true[i][0] - dic_est[i][0]) for i in range(num_players)]
    else:
        difference = [abs(dic_true[i][1] - dic_est[i][1]) for i in range(num_players)]
    return sum(difference) / num_players


def make_group(ranking: list, bin_size=400):
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
        if len(group) >= 2*n_teams:
            random.shuffle(group)
            team1 = group[:n_teams]
            team2 = group[n_teams:2*n_teams]
            pairs.append((team1, team2))
    return pairs


res = []
elos = [1500 for i in range(num_players)]
ranking = [[i, elos[i]] for i in range(num_players)]

for _ in range(num_sims):
    # Divide players into groups based on Elo ranges (e.g., bins of 200 points),
    # then match players within each group using simple methods (random)
    match_lis = make_group(ranking)
    for team_i, team_j in match_lis:
        mean_i = sum([elos[i] for i in team_i])/n_teams
        mean_j = sum([elos[j] for j in team_j])/n_teams
        e_i = 1 / (1 + 10 ** ((mean_j - mean_i) / sgm))
        e_j = 1 - e_i
        # assume no drawing
        true_mean_i = sum([true_elos[i] for i in team_i])/n_teams
        true_mean_j = sum([true_elos[j] for j in team_j])/n_teams
        s_i = np.random.binomial(n=1, p=1 / (1 + 10 ** ((true_mean_j - true_mean_i) / sgm)))
        s_j = 1 - s_i
        # update Elos in the same proportion as individual elo to the total elo within each team
        for i in team_i:
            elos[i] += K * (s_i - e_i) * elos[i] / (mean_i*n_teams)
        for j in team_j:
            elos[j] += K * (s_j - e_j) * elos[j] / (mean_j*n_teams)
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
plt.title(f"Team Game, Elo Metric, Group, {num_sims} Rounds, No Draw")
plt.xlabel("Iterations")
plt.ylabel("Metric")
plt.grid(True)
plt.ylim(0, 250)
plt.tight_layout()
plt.show()