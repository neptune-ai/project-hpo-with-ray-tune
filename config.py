import numpy as np
from ray import tune

config = {
    "lr": tune.loguniform(1e-4, 1e-1),
    "batch_size": tune.choice([2, 4, 8, 16, 32, 64, 128]),
    "momentum": tune.sample_from(lambda _: np.random.uniform(low=0.8, high=0.99)),
}
