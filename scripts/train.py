#!/usr/bin/env python3
import os, json, time, argparse
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", default="dataset")
parser.add_argument("--work_dir", default="models")
parser.add_argument("--epochs", type=int, default=40)
parser.add_argument("--batch", type=int, default=16)
parser.add_argument("--lr", type=float, default=2e-4)
parser.add_argument("--use_weights", action="store_true", help="use class weights")
args = parser.parse_args()

os.makedirs(args.work_dir, exist_ok=True)
IMG_SIZE = 224

train_tf = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8,1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])
val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE,IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

train_dir = os.path.join(args.data_dir, "train")
val_dir = os.path.join(args.data_dir, "val")
train_ds = datasets.ImageFolder(train_dir, transform=train_tf)
val_ds = datasets.ImageFolder(val_dir, transform=val_tf)

train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=0)

class_to_idx = train_ds.class_to_idx
json.dump(class_to_idx, open(os.path.join(args.work_dir, "class_to_idx.json"), "w"))
idx_to_class = {v:k for k,v in class_to_idx.items()}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(nn.Dropout(0.4), nn.Linear(num_ftrs, 512), nn.ReLU(), nn.Dropout(0.4), nn.Linear(512, len(class_to_idx)))
model = model.to(device)

# Optional class weights
if args.use_weights:
    targets = [y for _, y in train_ds.samples]
    classes, counts = np.unique(targets, return_counts=True)
    weights = {c: 1.0/counts[i] for i,c in enumerate(classes)}
    class_weights = torch.tensor([weights[i] for i in range(len(class_to_idx))], dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
else:
    criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=4)


best_val = 0.0
best_path = os.path.join(args.work_dir, "best_model.pth")

if __name__ == "__main__":
    for epoch in range(1, args.epochs+1):
        t0 = time.time()
        model.train()
        running_loss = 0.0; running_corrects = 0
        for imgs, labels in train_loader:
            imgs = imgs.to(device); labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
            running_corrects += (outputs.argmax(1) == labels).sum().item()
        train_loss = running_loss / len(train_ds)
        train_acc = running_corrects / len(train_ds)

        # validation
        model.eval()
        val_preds=[]; val_trues=[]
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                outputs = model(imgs)
                preds = outputs.argmax(1).cpu().numpy()
                val_preds.extend(preds.tolist())
                val_trues.extend(labels.numpy().tolist())
        val_acc = accuracy_score(val_trues, val_preds)
        scheduler.step(val_acc)
        print(f"Epoch {epoch}/{args.epochs} train_loss {train_loss:.4f} train_acc {train_acc:.4f} val_acc {val_acc:.4f} time {time.time()-t0:.1f}s")

        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), best_path)
            print("Saved best:", best_path)
    print("Training done. Best val acc:", best_val)
