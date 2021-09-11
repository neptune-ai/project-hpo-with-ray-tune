import os
from sklearn.metrics import accuracy_score
import torch.nn.functional as F
import neptune.new as neptune
import torch
import torch.nn as nn
import torch.optim as optim
from ray import tune
from torch.utils.data import random_split

from load_data import load_data, UnNormalize
from model import Net


def train_cifar(config, checkpoint_dir=None):
    # (neptune) create trial run
    trial_run = neptune.init(
        project="common/project-hpo-with-ray-tune",
        tags=["trial_run", "hpo_cifar"],
        name=tune.get_trial_id(),
    )

    # (neptune) log config
    trial_run["trial"]["config"] = config

    net = Net(config["l1"], config["l2"])

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            net = nn.DataParallel(net)
    net.to(device)

    # (neptune) log device
    trial_run["trial"]["device"] = device

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        net.parameters(),
        lr=config["lr"],
        momentum=config["momentum"]
    )

    if checkpoint_dir:
        checkpoint = os.path.join(checkpoint_dir, "checkpoint")
        model_state, optimizer_state = torch.load(checkpoint)
        net.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    data_dir = os.path.abspath("./data")
    train_set, test_set, classes = load_data(data_dir)

    test_abs = int(len(train_set) * 0.8)
    train_subset, val_subset = random_split(
        train_set,
        [test_abs, len(train_set) - test_abs]
    )

    train_loader = torch.utils.data.DataLoader(
        train_subset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=4
    )

    val_loader = torch.utils.data.DataLoader(
        val_subset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=4
    )

    un_normalize_img = UnNormalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

    for epoch in range(7):
        # Train
        tr_loss = 0.0
        tr_steps = 0
        tr_total = 0
        tr_correct = 0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = net(inputs)

            _, predicted = torch.max(outputs.data, 1)
            tr_total += labels.size(0)
            tr_correct += (predicted == labels).sum().item()

            loss = criterion(outputs, labels)

            # (neptune) log batch loss, have it as a chart in neptune
            trial_run["trial"]["metrics/train/batch/loss"].log(loss)

            y_true = labels.cpu().detach().numpy()
            y_pred = outputs.argmax(axis=1).cpu().detach().numpy()

            # (neptune) log batch accuracy
            trial_run["trial"]["metrics/train/batch/accuracy"].log(accuracy_score(y_true, y_pred))

            tr_loss += loss.cpu().detach().numpy()
            tr_steps += 1

            loss.backward()
            optimizer.step()

            # (neptune) log image predictions
            if i == len(train_loader) - 2:
                n_imgs = 0
                for image, label, prediction in zip(inputs, labels, outputs):
                    img = image.detach().cpu()
                    img_np = un_normalize_img(img).permute(1, 2, 0).numpy()

                    pred_label_idx = int(torch.argmax(F.softmax(prediction, dim=0)).numpy())

                    name = "pred: {}".format(classes[pred_label_idx])
                    desc_target = "target: {}".format(classes[label])
                    desc_classes = "\n".join(["class {}: {}".format(classes[i], pred)
                                              for i, pred in enumerate(F.softmax(prediction, dim=0))])
                    description = "{} \n{}".format(desc_target, desc_classes)
                    trial_run["trial"]["preds/train/epoch/{}".format(epoch)].log(
                        neptune.types.File.as_image(img_np),
                        name=name,
                        description=description
                    )
                    if n_imgs == 20:
                        break
                    n_imgs += 1

        # (neptune) log epoch loss and accuracy
        trial_run["trial"]["metrics/train/epoch/accuracy"].log(tr_correct / tr_total)
        trial_run["trial"]["metrics/train/epoch/loss"].log(tr_loss / tr_steps)

        # Valid
        val_loss = 0.0
        val_steps = 0
        total = 0
        correct = 0
        for i, data in enumerate(val_loader, 0):
            with torch.no_grad():
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                loss = criterion(outputs, labels)
                trial_run["trial"]["metrics/valid/batch/loss"].log(loss)

                y_true = labels.cpu().numpy()
                y_pred = outputs.argmax(axis=1).cpu().numpy()
                trial_run["trial"]["metrics/valid/batch/accuracy"].log(accuracy_score(y_true, y_pred))

                val_loss += loss.cpu().numpy()
                val_steps += 1

        trial_run["trial"]["metrics/valid/epoch/accuracy"].log(correct / total)
        trial_run["trial"]["metrics/valid/epoch/loss"].log(val_loss / val_steps)

        with tune.checkpoint_dir(step=epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save(
                (net.state_dict(), optimizer.state_dict()),
                path
            )
            trial_run["trial"]["checkpoint"].upload(path)

        trial_run.wait()
        tune.report(loss=(val_loss / val_steps), accuracy=correct / total)
