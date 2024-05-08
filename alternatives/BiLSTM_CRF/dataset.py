from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
from collections import Counter

# Load dataset
def get_conll2003_dataset():
    dataset = load_dataset("conll2003")
    #dataset = dataset.map(lambda example: {'tokens': example['tokens'], 'ner_tags': example['ner_tags']}, remove_columns=dataset.column_names)
    return dataset

# Example of a simple whitespace tokenizer
def simple_tokenizer(sentence):
    # alreay tokenized
    return sentence 

# Build vocabulary function
def build_vocab(texts, max_size=20000, min_freq=1):
    vocab_counter = Counter()
    for sentence in texts:
        tokens = simple_tokenizer(sentence)
        vocab_counter.update(tokens)
    
    sorted_vocab= sorted(vocab_counter.items(), key=lambda item: item[1], reverse=True)
    
    vocab = {word: idx + 2 for idx, (word, freq) in enumerate(sorted_vocab) if (freq >= min_freq) and (idx < max_size - 2)}
    vocab['<pad>'] = 0  # Add a padding token
    vocab['<unk>'] = 1  # Add an unknown token

    return vocab

# Convert tokens to IDs
def encode(text, vocab):
    encoded_sentence = [vocab.get(token, vocab['<unk>']) for token in simple_tokenizer(text)]
    return encoded_sentence

#max_len = max(len(simple_tokenizer(text)) for text in texts)

# Tokenize and encode the dataset
def process_data_123(examples):
    examples['input_ids'] = encode(examples['tokens'], vocab)
    #print(examples['input_ids'])
    return examples


# Collate function to create mini-batches
def collate_fn(batch):
    input_ids = [torch.tensor(item['input_ids']) for item in batch]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)  # Padding with the index of <pad>

    labels = [torch.tensor(item['ner_tags']) for item in batch]
    labels = pad_sequence(labels, batch_first=True, padding_value=-1) 

    mask = [torch.tensor(item['input_ids']) != 0 for item in batch]
    mask = pad_sequence(mask, batch_first=True, padding_value=False)

    return {"input_ids": input_ids, "labels": labels, "mask": mask}


vocab  = None
vocab_size = 0
label_len = 0
label_names = None

def init() :
    global vocab 
    global train_loader 
    global valid_loader 
    global test_loader 
    global vocab_size
    global label_len
    global label_names

    dataset = get_conll2003_dataset()
    label_names = dataset['train'].features['ner_tags'].feature.names

    train_texts = dataset['train']['tokens']
    valid_texts = dataset['validation']['tokens']
    # merge train and validation texts for build_vocab
    texts = train_texts + valid_texts

    vocab = build_vocab(texts)
    vocab_size = len(vocab)

    # Apply the processing function to the dataset
    tokenized_datasets = dataset.map(process_data_123)

    # DataLoader creation

    train_loader = DataLoader(tokenized_datasets['train'], batch_size=32, shuffle=False, collate_fn=collate_fn)
    valid_loader = DataLoader(tokenized_datasets['validation'], batch_size=32, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(tokenized_datasets['test'], batch_size=32, shuffle=False, collate_fn=collate_fn)

    label_len = len(dataset['train'].features['ner_tags'].feature.names)

def get_train_loader() :
    return train_loader

def get_valid_loader() :
    return valid_loader

def get_test_loader() :
    return test_loader


init()

if __name__ == "__main__" :
    train_loader = get_train_loader()
    for batch in train_loader:
        print("i", batch['input_ids'].shape)
        print("l", batch['labels'].shape)
