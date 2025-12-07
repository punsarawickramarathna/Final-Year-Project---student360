#!/usr/bin/env python3
import glob, subprocess, sys, collections, argparse, os
p = argparse.ArgumentParser()
p.add_argument("--model", default="models/best_model.pth")
p.add_argument("--class_map", default="models/class_to_idx.json")
p.add_argument("--faces_dir", default="sample_faces")
p.add_argument("--threshold", type=float, default=0.6)
args = p.parse_args()
faces = sorted(glob.glob(os.path.join(args.faces_dir,"*.jpg")))
counts = collections.Counter()
for f in faces:
    cmd = [sys.executable, "scripts/infer.py", "--model", args.model, "--class_map", args.class_map, "--img", f]
    res = subprocess.run(cmd, capture_output=True, text=True)
    out = res.stdout.strip().split()
    if len(out) >= 2:
        label = out[0]; conf = float(out[1])
        if conf >= args.threshold:
            counts[label] += 1
present = [k for k,v in counts.items() if v>=1]
print("Present students:", present)
print("Counts:", counts)
