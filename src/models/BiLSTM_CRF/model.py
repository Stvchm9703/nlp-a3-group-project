from torchcrf import CRF
import torch.nn as nn

import torch

class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_labels, dropout=0.1):
        super(BiLSTM_CRF, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers=1, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_labels)
        self.crf = CRF(num_labels, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, labels=None, mask=None):
        embeddings = self.embedding(input_ids)
        embeddings = self.dropout(embeddings)
        outputs, _ = self.lstm(embeddings)
        emissions = self.fc(outputs)
        if labels is not None:
            # Training: return the loss
            return self.compute_loss(emissions, labels, mask)
        else:
            # Inference: return the decoded labels
            return self.crf.decode(emissions, mask=mask)

    def compute_loss(self, emissions, labels, mask):
        """
        Compute the CRF loss.
        
        Args:
            emissions (torch.Tensor): Emission values from the linear layer, shape [batch_size, seq_length, num_labels]
            labels (torch.Tensor): Ground-truth labels, shape [batch_size, seq_length]
            mask (torch.Tensor): Mask tensor to ignore the padded items in the sequences, shape [batch_size, seq_length]

        Returns:
            torch.Tensor: The loss value.
        """
        # The CRF layer uses negative log likelihood loss
        loss = -self.crf(emissions, labels, mask=mask, reduction='mean')
        return loss

# Example initialization

class BiLSTM_CRF_bak(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_labels, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers=1, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_labels)
        self.crf = CRF(num_labels, batch_first=True)
        
    def forward(self, input_ids, labels=None, mask=None):
        embeddings = self.embedding(input_ids)
        outputs, _ = self.lstm(embeddings)
        emissions = self.fc(outputs)
        if labels is not None:
            # mask를 사용하여 CRF 계산
            #print("emissions.shape", emissions.shape)
            #print("labels.shape:", labels.shape)
            #print("mask.shape:", mask.shape)
            #print("emissions[0]", emissions[0])
            #print("labels[0]", labels[0])
            loss = -self.crf(emissions, labels, mask=mask, reduction='mean')
            return loss
        else:
            # 예측 시에도 mask 적용
            predictions = self.crf.decode(emissions, mask=mask)
            return predictions


def create_model(vocab_size, label_len):
    model = BiLSTM_CRF(vocab_size, 128, 256, label_len)

    return model
