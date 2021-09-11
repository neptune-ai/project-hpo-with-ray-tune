# based on:
# https://github.com/ray-project/ray/blob/master/python/ray/tune/examples/cifar10_pytorch.py
# accessed 2021.09.11

import neptune.new as neptune
from ray import tune

from config import config
from main_train import train_cifar
from test_best_model import test_best_model

name = "hpo_cifar"

# (neptune) create run
master_run = neptune.init(
    project="common/project-hpo-with-ray-tune",
    tags=["master_run", name],
)


def main(cfg, num_samples):
    result = tune.run(
        tune.with_parameters(train_cifar),
        resources_per_trial={"cpu": 4},
        config=cfg,
        metric="loss",
        mode="min",
        num_samples=num_samples,
        name=name,
        keep_checkpoints_num=1,
        checkpoint_score_attr="min-loss",
    )

    # (neptune) log best trial metadata
    master_run["best/config"] = result.best_config
    master_run["best/dataframe"].upload(neptune.types.File.as_html(result.best_dataframe))
    master_run["best/result"] = result.best_result
    master_run["best/trial"] = result.best_trial

    best_trial = result.get_best_trial("loss", "min", "last")
    test_best_model(master_run, best_trial)


if __name__ == "__main__":
    main(cfg=config, num_samples=2)
