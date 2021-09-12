# based on:
# https://github.com/ray-project/ray/blob/master/python/ray/tune/examples/cifar10_pytorch.py
# accessed 2021.09.11

import neptune.new as neptune
import numpy as np
from ray import tune
from ray.tune.schedulers import PopulationBasedTraining

from config import config
from main_train import train_cifar


class CustomStopper(tune.Stopper):
    def __init__(self):
        self.should_stop = False

    def __call__(self, trial_id, result):
        max_iter = 100
        if not self.should_stop and result["loss"] < 0.01:
            self.should_stop = True
        return self.should_stop or result["training_iteration"] >= max_iter

    def stop_all(self):
        return self.should_stop


name = "pbt"

# (neptune) create run
master_run = neptune.init(
    project="common/project-hpo-with-ray-tune",
    tags=["master_run", name],
)


def main(cfg):
    result = tune.run(
        tune.with_parameters(train_cifar),
        resources_per_trial={"cpu": 4},
        scheduler=PopulationBasedTraining(
            time_attr="training_iteration",
            metric="loss",
            mode="min",
            perturbation_interval=3,
            resample_probability=0.3,
            hyperparam_mutations={
                "lr": tune.loguniform(1e-4, 1e-1),
                "batch_size": [2, 4, 8, 16, 32, 64, 128],
                "momentum": lambda: np.random.uniform(low=0.8, high=0.99),
            },
        ),
        config=cfg,
        num_samples=5,
        name=name,
        stop=CustomStopper(),
        keep_checkpoints_num=5,
        checkpoint_score_attr="min-loss",
    )

    # (neptune) log best trial metadata
    best_trial = result.get_best_trial("loss", "min")
    master_run["best/config"] = best_trial.config
    master_run["best/trial"] = best_trial.trial_id


if __name__ == "__main__":
    main(cfg=config)
