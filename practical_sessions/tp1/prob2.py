import numpy as np

rng = np.random.default_rng()

n = int(10e6)

X = rng.binomial(20, 0.2, n) + 1
Y = np.array([rng.binomial(3 ** x, 0.5) for x in X])

pred = (3 ** X) / 2

loss = np.square(Y - pred)
empirical = loss.mean()

print(empirical)
