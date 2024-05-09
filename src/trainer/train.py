# from .. import (models , dataloaders)

from models.transformer import NERClassifier
from dataloaders.CoNLLDataset import (
    create_train_dataset,
    create_validation_dataset,
    create_test_dataset,
)
from poutyne import (
    set_seeds,
    Model as ModelTrainner,
    ModelCheckpoint,
    CSVLogger,
    ModelBundle,
)

import time
from datetime import datetime
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import optim
from sklearn.metrics import accuracy_score, f1_score, classification_report

import pprint

# from . import util


def train_main(config, model, **kwargs):
    now_time = f"{datetime.now():%Y-%m-%d_%H-%M-%S}"
    set_seeds(20240510)
    model_name = kwargs.get("model_name")
    print("start for " + model_name)
    cuda_device = 0
    device = torch.device(
        "cuda:%d" % cuda_device if torch.cuda.is_available() else "cpu"
    )

    batch = config["batch_size"]["train"]
    learning_rate = config["train_config"]["learning_rate"]

    epochs = config["train_config"]["num_of_epochs"]
    l2_penalty = config["train_config"]["l2_penalty"]
    gradient_clipping = config["train_config"]["gradient_clipping"]

    train_loader = create_train_dataset(config)
    valid_loader = create_validation_dataset(config)
    test_loader = create_test_dataset(config)

    class_w = config["train_config"]["class_w"]
    class_w = torch.tensor(class_w).to(device)
    class_w /= class_w.sum()

    # Prepare the model optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=l2_penalty,
    )

    loss_function = torch.nn.CrossEntropyLoss(weight=class_w)
    # loss_function = torch.nn.MultiLabelSoftMarginLoss(weight=class_w)
    os.makedirs(
        os.path.join(
            "checkpoints",
            model_name,
            now_time,
        ),
        exist_ok=True,
    )
    os.makedirs(os.path.join("logs"), exist_ok=True)

    callbacks = [
        # Save the latest weights to be able to continue the optimization at the end for more epochs.
        ModelCheckpoint(
            os.path.join("checkpoints", model_name, now_time, "epoch_{}.ckpt")
        ),
        # Save the weights in a new file when the current model is better than all previous models.
        ModelCheckpoint(
            os.path.join(
                "checkpoints", model_name, now_time, "best_epoch_{epoch}.ckpt"
            ),
            monitor="val_acc",
            mode="max",
            save_best_only=True,
            restore_best=True,
            verbose=True,
        ),
        # Save the losses and accuracies for each epoch in a TSV.
        CSVLogger(
            os.path.join("logs", f"training__{model_name}__{now_time}.tsv"),
            separator="\t",
        ),
    ]

    # Poutyne Model on GPU
    model_trainer = ModelTrainner(
        model,
        optimizer,
        loss_function,
        batch_metrics=["accuracy"],
        device=device,
    )

    # Train
    model_trainer.fit_generator(
        train_loader, valid_loader, epochs=epochs, callbacks=callbacks
    )

    # Test
    test_loss, test_acc = model_trainer.evaluate_generator(test_loader)

    pprint({"test_loss": test_loss, "test_acc": test_acc})
