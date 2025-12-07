#!/usr/bin/env python3
import os, cv2, argparse

def save_crop(img, box, out_path, size=(160,160)):
    x,y,w,h = box
    x,y = max(0,int(x)), max(0,int(y))
    crop = img[y:y+int(h), x:x+int(w)]
    if crop.size == 0:
        return False
    crop = cv2.resize(crop, size)
    cv2.imwrite(out_path, crop)
    return True

def extract_from_image(path, out_folder, prefix="img"):
    os.makedirs(out_folder, exist_ok=True)
    img = cv2.imread(path)
    if img is None:
        print("Could not load image:", path)
        return 0

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    saved = 0
    for i,(x,y,w,h) in enumerate(faces):
        if save_crop(img, (x,y,w,h), os.path.join(out_folder, f"{prefix}_{i}.jpg")):
            saved += 1
    return saved

def extract_from_video(video_path, out_folder, frame_step=10):
    os.makedirs(out_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    idx = 0
    saved = 0

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if idx % frame_step == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            for i,(x,y,w,h) in enumerate(faces):
                if save_crop(frame, (x,y,w,h), os.path.join(out_folder, f"frame{idx:06d}_face{i}.jpg")):
                    saved += 1

        idx += 1

    cap.release()
    return saved

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--video", action="store_true")
    p.add_argument("--frame_step", type=int, default=10)
    args = p.parse_args()

    if args.video:
        n = extract_from_video(args.input, args.out, args.frame_step)
    else:
        n = extract_from_image(args.input, args.out)

    print("Saved faces:", n)
