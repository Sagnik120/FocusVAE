"""
Download and subset the CelebA dataset via the Kaggle API.

Requires a Kaggle API token at ~/.kaggle/kaggle.json
(https://www.kaggle.com/docs/api -> "Create New Token").

Usage:
    python data/download_celeba.py --out data/celeba --subset 15000
"""
import argparse
import shutil
import zipfile
from pathlib import Path


KAGGLE_DATASET = "jessicali9530/celeba-dataset"


def download(out_dir: Path):
    import kaggle  # imported lazily so --help works without credentials set

    out_dir.mkdir(parents=True, exist_ok=True)
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(KAGGLE_DATASET, path=str(out_dir), quiet=False)

    zip_path = out_dir / "celeba-dataset.zip"
    if zip_path.exists():
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(out_dir)
        zip_path.unlink()


def subset(out_dir: Path, n: int):
    """Keep only the first `n` images (by filename order) plus all metadata."""
    img_dir = out_dir / "img_align_celeba" / "img_align_celeba"
    if not img_dir.exists():
        # some kaggle mirrors nest one level deeper/shallower
        candidates = list(out_dir.rglob("img_align_celeba"))
        img_dir = next((c for c in candidates if c.is_dir() and any(c.glob("*.jpg"))), None)
        if img_dir is None:
            raise FileNotFoundError("Could not locate img_align_celeba directory after extraction")

    all_imgs = sorted(img_dir.glob("*.jpg"))
    keep = set(p.name for p in all_imgs[:n])

    subset_dir = out_dir / "images"
    subset_dir.mkdir(exist_ok=True)
    for p in all_imgs:
        if p.name in keep:
            shutil.copy(p, subset_dir / p.name)

    print(f"Kept {len(keep)} / {len(all_imgs)} images in {subset_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="data/celeba")
    ap.add_argument("--subset", type=int, default=15000,
                     help="Number of images to keep (lean training on M-series Macs)")
    ap.add_argument("--skip_download", action="store_true",
                     help="Skip download if you already extracted the Kaggle zip manually")
    args = ap.parse_args()

    out_dir = Path(args.out)
    if not args.skip_download:
        download(out_dir)
    subset(out_dir, args.subset)


if __name__ == "__main__":
    main()
