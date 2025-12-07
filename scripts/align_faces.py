# scripts/align_faces.py
import os
import argparse
from facenet_pytorch import MTCNN
from PIL import Image

def align_all(src_dir, dst_dir, size=160):
    os.makedirs(dst_dir, exist_ok=True)
    detector = MTCNN(image_size=size, margin=20)

    total = 0
    for fname in os.listdir(src_dir):
        path = os.path.join(src_dir, fname)

        try:
            img = Image.open(path).convert("RGB")
        except:
            print("Cannot open:", path)
            continue

        face = detector(img)
        if face is None:
            print("No face detected:", fname)
            continue

        out_path = os.path.join(dst_dir, fname)
        face_img = face.permute(1, 2, 0).numpy()  # CHW → HWC
        Image.fromarray((face_img * 255).astype("uint8")).save(out_path)
        total += 1

    print("Total aligned:", total)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True)
    p.add_argument("--dst", required=True)
    args = p.parse_args()

    align_all(args.src, args.dst)
