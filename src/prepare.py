import json
import pprint
from dataloaders.util import download_dataset, create_vocabulary, extract_embeddings


if __name__ == '__main__' :
    config_path = "config.json"
    with open(config_path, "r") as f:
        config = json.load(f)

    train_set, _, _ = download_dataset(config["dataset_dir"])
    vocab = create_vocabulary(train_set, config["vocab_size"])
    # Extract GloVe embeddings for tokens present in the training set vocab
    pprint.pprint(vocab)

    extract_embeddings(config, vocab)
