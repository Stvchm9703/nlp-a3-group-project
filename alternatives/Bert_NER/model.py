# model.py
from transformers import BertForTokenClassification

def create_model(vocab_size, label_len):
    model = BertForTokenClassification.from_pretrained('bert-base-cased', num_labels=label_len)
    return model 
