from ray import tune

config = {
    "lr": tune.loguniform(1e-4, 1e-1),
    "batch_size": tune.choice([2, 4, 8, 16, 32, 64, 128]),
    "momentum": tune.choice([0.8, 0.85, 0.9, 0.95, 0.99]),
}
