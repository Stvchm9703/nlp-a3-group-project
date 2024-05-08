import json

import torch
# from tensorboardX import SummaryWriter

from models.NerTransformer.classifier import NERClassifier
from  trainer import train_loop
from trainer import create_trainner
from torch.utils.data import DataLoader

from dataloaders import CoNLLDataset

def main():
    # Load the pipeline configuration file
    with open("config.json", "r", encoding="utf8") as f:
        config = json.load(f)

    # writer = SummaryWriter()
    use_gpu = config["use_gpu"] and torch.cuda.is_available()
    device = torch.device("cuda" if use_gpu else "cpu")

    # train_loop(config, writer, device)
    model = NERClassifier(config).to(device)
    # train_set = CoNLLDataset(config, "train")
    # valid_set = CoNLLDataset(config, "validation")
    train_loader = DataLoader(CoNLLDataset(config, "train"),
                              batch_size=config["batch_size"]["train"],
                              shuffle=True,
                              drop_last=True)
    valid_loader = DataLoader(CoNLLDataset(config, "validation"),
                              batch_size=config["batch_size"]["validation"],
                              shuffle=False,
                              drop_last=True)
    create_trainner(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=valid_loader,
        config=config['trainer'],
        device=device)


if __name__ == "__main__":
    main()
