# sort_tiny_val.py
import argparse, os, shutil, pandas as pd
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="tiny-imagenet-200 根目录")
    ap.add_argument("--dest", default="val_sorted", help="输出目录名（位于 root/ 下）")
    ap.add_argument("--mode", choices=["copy","move"], default="copy", help="copy 更安全，move 更省空间")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    val_dir = root / "val"
    anno = val_dir / "val_annotations.txt"
    img_dir = val_dir / "images"
    out_dir = root / args.dest
    out_dir.mkdir(exist_ok=True)

    # 读取标注
    df = pd.read_csv(anno, sep="\t", header=None,
                     names=["filename","wnid","x1","y1","x2","y2"])
    print(f"[INFO] annotations: {len(df)} rows")

    # 分发
    n_ok, n_miss = 0, 0
    for _, r in df.iterrows():
        src = img_dir / r["filename"]
        dst_class = out_dir / r["wnid"]
        dst_class.mkdir(parents=True, exist_ok=True)
        dst = dst_class / r["filename"]
        if not src.exists():
            n_miss += 1
            continue
        if args.mode == "copy":
            shutil.copy2(src, dst)
        else:
            shutil.move(src, dst)
        n_ok += 1

    print(f"[DONE] {args.mode}: {n_ok} files. missing: {n_miss}.")
    if args.mode == "move":
        try:
            os.rmdir(img_dir)  # 为空则删除
            print(f"[CLEAN] removed empty dir: {img_dir}")
        except OSError:
            print(f"[CLEAN] {img_dir} not empty or already removed; skip.")

if __name__ == "__main__":
    print('1')
    main()
