import numpy as np
from ray import tune

config = {
    "l1": tune.sample_from(lambda _: 2 ** np.random.randint(2, 11)),
    "l2": tune.sample_from(lambda _: 2 ** np.random.randint(2, 11)),
    "lr": tune.loguniform(1e-4, 1e-1),
    "batch_size": tune.choice([2, 4, 8, 16, 32, 64, 128]),
    "momentum": tune.choice([0.9, 0.95, 0.99]),
}
