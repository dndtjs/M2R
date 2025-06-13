import numpy as np
import matplotlib.pyplot as plt

num_players = 100
num_sims = 1000

true_elos = np.random.normal(1500, 300, num_players)
true_ranking = [[i, true_elos[i]] for i in range(num_players)]
true_ranking.sort(key=lambda x: x[1], reverse=True)
dic_true = {true_ranking[i][0]: (i + 1, true_ranking[i][1]) for i in range(num_players)}

def metric(dic_true, dic_est, rank=False):
    if rank:
        difference = [abs(dic_true[i][0] - dic_est[i][0]) for i in range(num_players)]
    else:
        difference = [abs(dic_true[i][1] - dic_est[i][1]) for i in range(num_players)]
    return sum(difference) / len(difference)

res = []
elos = [1500 for i in range(num_players)]
ranking = [[i, elos[i]] for i in range(num_players)]

for n in range(num_sims):
    if n > 30:
        K = 20
    else:
        K = 40
    for k in range(0, num_players, 2):
        i, j = ranking[k][0], ranking[k + 1][0]
        e_i = 1 / (1 + 10 ** ((elos[j] - elos[i]) / 400))
        e_j = 1 - e_i
        s_i = np.random.binomial(n=1, p=1 / (1 + 10 ** ((true_elos[j] - true_elos[i]) / 400)))
        s_j = 1 - s_i
        K_i = K_j = K
        if elos[i] > 2400 or elos[i] < 600:
            K_i = 10
        if elos[j] > 2400 or elos[i] < 600:
            K_j = 10
        elos[i] += K_i * (s_i - e_i)
        elos[j] += K_j * (s_j - e_j)
    ranking = [[i, elos[i]] for i in range(num_players)]
    ranking.sort(key=lambda x: x[1], reverse=True)
    dic_est = {ranking[i][0]: (i + 1, ranking[i][1]) for i in range(num_players)}
    res.append(metric(dic_true, dic_est))

print(dic_true)
print("Estimate:")
print(dic_est)

plt.plot(range(len(res)), res)
plt.title(f"Elo Metric, Rivalry, {num_sims} Rounds, No Draw")
plt.xlabel("Number of Rounds")
plt.ylabel("Mean Elo Error")
plt.grid(True)
plt.tight_layout()
plt.show()
