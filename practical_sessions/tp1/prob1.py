import numpy as np

rng = np.random.default_rng()

n = int(10e5)

X = np.random.randint(0, high = 2, size = n)
Y = np.array([rng.binomial(1, 0.6) if x == 1 else rng.binomial(1, 0.4) for x in X])

#Prediction with random
pred = X

#Computing empirical risk with the given loss
loss = np.abs(Y - pred)
empirical = loss.mean() # loss.sum() / n

print(empirical)
