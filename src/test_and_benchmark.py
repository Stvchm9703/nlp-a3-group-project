import torch
from torch.utils.data import DataLoader
from torchsummary import summary
import argparse
import dataloaders.CoNLLDataset as CoNLLDataset
from trainer.test import evaluate
from models import NERClassifier
import json
import pprint


def run_test(model, config):
    # This is a test function
    print("Running test function")
    test_set = CoNLLDataset(config, "test")
    test_loader = DataLoader(
        test_set,
        batch_size=config["batch_size"]["test"],
        shuffle=True,
        drop_last=True,
    )

    evaluate(model, test_loader)


def main():
    parser = argparse.ArgumentParser(description="Test and Benchmark")
    parser.add_argument(
        "--model_path", type=str, default="model.pth", help="Path to the model file"
    )
    parser.add_argument(
        "--config", type=str, default="config.json", help="Path to the config file"
    )
    args = parser.parse_args()

    cfg_str = open(args.config, "r", encoding="utf-8").read()
    cfg = json.loads(cfg_str)
    # Rest of the code

    checkpoint = torch.load(args.model_path)
    model = NERClassifier(cfg)
    # pprint([key for key, value in enum checkpoint.items()])
   

    model.load_state_dict(checkpoint)

    run_test(model, cfg)


if __name__ == "__main__":
    main()
