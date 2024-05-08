from torch.optim import Adam
from .model import create_model
from .dataset import get_train_loader, get_valid_loader, label_names
from tqdm import tqdm
from sklearn.metrics import classification_report
import torch

def train_epoch(model, data_loader, optimizer, epoch):
    model.train()
    total_loss = 0
    progress_bar = tqdm(enumerate(data_loader), total=len(data_loader), desc=f"Train Epoch {epoch}")
    for i, batch in progress_bar:
    #for batch in data_loader:
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        mask = batch['mask'].to(device)  # mask를 배치에서 가져옵니다.

        optimizer.zero_grad()
        loss = model(input_ids, labels, mask)  # mask 인자를 포함하여 모델을 호출합니다.
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(data_loader)

def eval_model(model, data_loader, mode):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        progress_bar = tqdm(enumerate(data_loader), total=len(data_loader), desc=f"Eval {mode}")
        for i, batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            mask = batch['mask'].to(device)

            # The forward method should now only return the loss when labels are provided
            loss = model(input_ids, labels=labels, mask=mask)
            total_loss += loss.item()

            # For predictions, use the model without labels to invoke the decode method in the CRF
            preds = model(input_ids, mask=mask)  # This should now call the decode method

            # Flatten labels for evaluation metrics if needed outside CRF context
            labels_flat = labels[mask.bool()].tolist()  # Mask applied and flattened
            preds_flat = [p for pred, m in zip(preds, mask.bool().tolist()) for p, m in zip(pred, m) if m]

            all_preds.extend(preds_flat)
            all_labels.extend(labels_flat)

    # After the loop, you can compute metrics such as accuracy, F1 score, etc.
    # These are available via sklearn's classification report or other metric functions
    print(classification_report(all_labels, all_preds, target_names=label_names, zero_division=0))

    return total_loss / len(data_loader)

train_loader = get_train_loader()
valid_loader = get_valid_loader()

# 모델 훈련
device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
print(f"Using device: {device}")

model = create_model(20000, 9)
print(model)
model = model.to(device)
optimizer = Adam(model.parameters(), lr=5e-5)

for epoch in range(20):  
    train_loss = train_epoch(model, train_loader, optimizer, epoch)
    train_eval_loss = eval_model(model, train_loader, "Train")
    val_loss = eval_model(model, valid_loader, "Validation")
    print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Train Eval Loss = {train_eval_loss:.4f}, Validation Loss = {val_loss:.4f}")

