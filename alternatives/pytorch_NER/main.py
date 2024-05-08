import json

import torch
from tensorboardX import SummaryWriter

from trainer import train_loop


def main():
    # Load the pipeline configuration file
    with open("./config.json", "r", encoding="utf8") as f:
        config = json.load(f)

    writer = SummaryWriter()
    use_gpu = config["use_gpu"] and torch.cuda.is_available()
    #device = torch.device("cuda" if use_gpu else "cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"Using device: {device}")

    train_loop(config, writer, device)


if __name__ == "__main__":
    main()
