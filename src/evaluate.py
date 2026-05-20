import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from torch.utils.data import DataLoader

from config import *
from dataset import TumorDataset
from model import BrainTumorCNN
from train import class_to_idx, idx_to_class, test_samples


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


checkpoint = torch.load(MODEL_DIR / "best_model.pth")

model = BrainTumorCNN(num_classes=len(class_to_idx))
model.load_state_dict(checkpoint["model_state_dict"])
model.to(DEVICE)

model.eval()


test_dataset = TumorDataset(test_samples, class_to_idx, use_crop=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)


all_preds = []
all_labels = []


with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE)

        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())


accuracy = accuracy_score(all_labels, all_preds)

print(f"Accuracy: {accuracy:.4f}")

print(classification_report(
    all_labels,
    all_preds,
    target_names=class_to_idx.keys()
))


cm = confusion_matrix(all_labels, all_preds)


plt.figure(figsize=(15, 12))
sns.heat