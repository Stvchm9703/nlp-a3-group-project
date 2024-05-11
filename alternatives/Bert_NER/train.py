from torch.optim import Adam
from .model import create_model
from ..BiLSTM_CRF.dataset import label_len, label_names
from .dataset import get_train_loader, get_valid_loader, get_test_loader #, get_sampled_train_loader, get_sampled_valid_loader
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import torch

def train_epoch(model, data_loader, optimizer, epoch):
    model.train()
    total_loss = 0
    progress_bar = tqdm(enumerate(data_loader), total=len(data_loader), desc=f"Train Epoch {epoch}")
    for i, batch in progress_bar:
    #for batch in data_loader:
        batch = {k: v.to(device) for k, v in batch.items()}

        optimizer.zero_grad()
        output = model(**batch)
        loss = output.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(data_loader)


from collections import Counter

def eval_model(model, data_loader, mode):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        progress_bar = tqdm(enumerate(data_loader), total=len(data_loader), desc=f"Eval {mode}")
        for i, batch in progress_bar:
            batch = {k: v.to(device) for k, v in batch.items()}

            labels = batch['labels']
            outputs = model(**batch)
            loss = outputs.loss
            logits = outputs.logits

            total_loss += loss.item()

            # Decode logits to predictions
            preds = torch.argmax(logits, dim=2)

            # Flatten the outputs
            mask = batch['attention_mask'].bool()
            labels_flat = labels[mask].tolist()
            preds_flat = preds[mask].tolist()

            all_preds.extend(preds_flat)
            all_labels.extend(labels_flat)


    # -100 레이블 제거
    filtered_labels = [label for label in all_labels if label != -100]
    filtered_preds = [pred for label, pred in zip(all_labels, all_preds) if label != -100]

    if mode=="Test":
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(10,7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_names, yticklabels=label_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.savefig('images/Bert_NER_confusion_matrix.png')  # Save as PNG
        plt.close()
    else:
        print(classification_report(filtered_labels, filtered_preds, target_names=label_names, zero_division=0))


    return total_loss / len(data_loader)


train_loader = get_train_loader()
valid_loader = get_valid_loader()
test_loader = get_test_loader()

#train_loader = get_sampled_train_loader()
#valid_loader = get_valid_loader()

# 모델 훈련
device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
print(f"Using device: {device}")
model = create_model(20000, 9)

print(model)
model = model.to(device)
optimizer = Adam(model.parameters(), lr=5e-5)

#epoch_num = 0
epoch_num = 20
for epoch in range(epoch_num):  
    train_loss = train_epoch(model, train_loader, optimizer, epoch)
    train_eval_loss = eval_model(model, train_loader, "Train")
    val_loss = eval_model(model, valid_loader, "Validation")
    #print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
    print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Train Eval Loss = {train_eval_loss:.4f}, Validation Loss = {val_loss:.4f}")

eval_model(model, test_loader, "Test")
