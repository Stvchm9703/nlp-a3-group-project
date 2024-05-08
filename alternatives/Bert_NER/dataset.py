# dataset.py
from datasets import load_dataset
from transformers import BertTokenizerFast
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch

# 데이터셋 로드
dataset = load_dataset("conll2003")

# 토크나이저 로드
tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')

def tokenize_and_align_labels(examples):
    # truncation과 padding을 True로 설정
    tokenized_inputs = tokenizer(examples['tokens'], truncation=True, 
                                 padding=True, is_split_into_words=True)

    labels = []
    for i, label in enumerate(examples['ner_tags']):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)  # Special token
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)  # Padding
            previous_word_idx = word_idx
        labels.append(label_ids)
    
    tokenized_inputs['labels'] = labels
    return tokenized_inputs

# 데이터셋 토큰화
tokenized_datasets = dataset.map(tokenize_and_align_labels, batched=True)
#tokenized_datasets = dataset.map(tokenize_and_align_labels, batched=True, load_from_cache_file=False)


# 사용자 정의 collate_fn 함수
def collate_fn(batch):
    input_ids = [item['input_ids'] for item in batch]
    labels = [item['labels'] for item in batch]

    # pad_sequence를 사용하여 각 시퀀스를 배치의 최대 길이에 맞춰 패딩
    input_ids_padded = pad_sequence([torch.tensor(seq) for seq in input_ids],
                                    batch_first=True, padding_value=tokenizer.pad_token_id)
    labels_padded = pad_sequence([torch.tensor(lab) for lab in labels],
                                 batch_first=True, padding_value=-100)

    # attention_mask 생성
    attention_mask = (input_ids_padded != tokenizer.pad_token_id).int()

    return {'input_ids': input_ids_padded, 'attention_mask': attention_mask, 'labels': labels_padded}

# DataLoader에 collate_fn 설정 추가
train_loader = DataLoader(tokenized_datasets['train'], batch_size=32, shuffle=True, collate_fn=collate_fn)
valid_loader = DataLoader(tokenized_datasets['validation'], batch_size=32, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(tokenized_datasets['test'], batch_size=32, shuffle=False, collate_fn=collate_fn)

def get_train_loader():
    return train_loader

def get_valid_loader():
    return valid_loader

def get_test_loader():
    return test_loader


# 각 분할의 10%만큼 샘플을 선택
sampled_train = tokenized_datasets['train'].select(range(0, len(tokenized_datasets['train']), 10))
sampled_validation = tokenized_datasets['validation'].select(range(0, len(tokenized_datasets['validation']), 10))
sampled_test = tokenized_datasets['test'].select(range(0, len(tokenized_datasets['test']), 10))

sampled_train_loader = DataLoader(sampled_train, batch_size=32, shuffle=True, collate_fn=collate_fn)
sampled_valid_loader = DataLoader(sampled_validation, batch_size=32, shuffle=False, collate_fn=collate_fn)
sampled_test_loader = DataLoader(sampled_test, batch_size=32, shuffle=False, collate_fn=collate_fn)

def get_sampled_train_loader():
    return sampled_train_loader

def get_sampled_valid_loader():
    return sampled_valid_loader

def get_sampled_test_loader():
    return sampled_test_loader


if __name__ == "__main__":
    print(len(train_loader))
    print(len(sampled_train_loader))

    for batch in sampled_train_loader:
        labels = batch['labels']
        print(labels.shape)
#    print(labels)
#    break;
