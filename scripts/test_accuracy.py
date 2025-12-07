#!/usr/bin/env python3
import os, json, argparse
import torch, numpy as np
from torchvision import transforms, models
from sklearn.metrics import classification_report, confusion_matrix
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", required=True)
parser.add_argument("--model", default="models/best_model.pth")
parser.add_argument("--class_map", default="models/class_to_idx.json")
args = parser.parse_args()

class_map = json.load(open(args.class_map))
idx_to_class = {int(v):k for k,v in class_map.items()}
num_classes = len(class_map)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet34(pretrained=False)
model.fc = torch.nn.Sequential(torch.nn.Dropout(0.4), torch.nn.Linear(model.fc.in_features, 512), torch.nn.ReLU(), torch.nn.Dropout(0.4), torch.nn.Linear(512, num_classes))
model.load_state_dict(torch.load(args.model, map_location=device))
model.to(device); model.eval()

tf = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

y_true = []; y_pred = []; files = []
for cls in sorted(os.listdir(args.data_dir)):
    cls_dir = os.path.join(args.data_dir, cls)
    if not os.path.isdir(cls_dir): continue
    for f in sorted(os.listdir(cls_dir)):
        if not f.lower().endswith(('.jpg','.png','.jpeg')): continue
        p = os.path.join(cls_dir, f)
        img = Image.open(p).convert("RGB")
        x = tf(img).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(x)
            pred = int(out.argmax(dim=1).cpu().numpy()[0])
        y_true.append(cls); y_pred.append(idx_to_class[pred]); files.append(p)

print("Classification report:")
print(classification_report(y_true, y_pred, digits=4))
cm = confusion_matrix(y_true, y_pred, labels=sorted(class_map.keys()))
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=sorted(class_map.keys()), yticklabels=sorted(class_map.keys()))
plt.xlabel("Predicted"); plt.ylabel("True")
plt.show()
