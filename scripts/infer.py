#!/usr/bin/env python3
import argparse, json, cv2, torch
from torchvision import transforms, models
from PIL import Image
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--model", default="models/best_model.pth")
p.add_argument("--class_map", default="models/class_to_idx.json")
p.add_argument("--img", required=True)
p.add_argument("--threshold", type=float, default=0.75)
args = p.parse_args()

class_map = json.load(open(args.class_map))
inv = {int(v):k for k,v in class_map.items()}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet34(pretrained=False)
model.fc = torch.nn.Sequential(torch.nn.Dropout(0.4), torch.nn.Linear(model.fc.in_features, 512), torch.nn.ReLU(), torch.nn.Dropout(0.4), torch.nn.Linear(512, len(class_map)))
model.load_state_dict(torch.load(args.model, map_location=device))
model.to(device); model.eval()

def preprocess(path):
    img = cv2.imread(path)
    if img is None:
        raise ValueError("Could not load image")
    # optional: histogram equalization on Y channel
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    img_eq = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    pil = Image.fromarray(cv2.cvtColor(img_eq, cv2.COLOR_BGR2RGB))
    return pil

transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

img = preprocess(args.img)
x = transform(img).unsqueeze(0).to(device)
with torch.no_grad():
    out = model(x)
    probs = torch.nn.functional.softmax(out, dim=1).cpu().numpy()[0]
    idx = int(probs.argmax())
    conf = float(probs[idx])
label = inv[idx] if conf >= args.threshold else "unknown"
print(label, conf)
