import os

import neptune.new as neptune
import torch

from load_data import load_data
from model import Net


def test_best_model(run, best_trial):
    best_trained_model = Net(best_trial.config["l1"], best_trial.config["l2"])
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    best_trained_model.to(device)

    # (neptune) fetch project
    project = neptune.get_project(name="common/project-hpo-with-ray-tune")

    # (neptune) find best trial
    best_run_df = project.fetch_runs_table(owner="kamil", tag="trial_run").to_pandas()
    best_run_df = best_run_df.sort_values(by=["trial/metrics/valid/epoch/loss"])
    best_run_id = best_run_df["sys/id"].values[0]

    # (neptune) resume this run
    best_run = neptune.init(
        project="common/project-hpo-with-ray-tune",
        run=best_run_id,
        mode="read-only",
    )

    # (neptune) download model and close run
    checkpoint_path = "./best_checkpoint"
    best_run["trial/checkpoint"].download(checkpoint_path)
    best_run.stop()

    model_state, optimizer_state = torch.load(checkpoint_path)
    best_trained_model.load_state_dict(model_state)

    _, test_set, _ = load_data(
        os.path.abspath("/abs/path/to/data")
    )

    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=4, shuffle=False, num_workers=2)

    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = best_trained_model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # (neptune) log test accuracy
    run["best/test_accuracy"] = correct / total
