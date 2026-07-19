import argparse
import random
import sys
from pathlib import Path
import cv2
import yaml

# Add root folder to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))


def validate_yolo_dataset(data_yaml_path: Path):
    """
    Validates data.yaml structure, image-label pairing, and class IDs.
    """
    if not data_yaml_path.exists():
        raise FileNotFoundError(f"data.yaml not found at: {data_yaml_path}")

    # Load data.yaml
    with open(data_yaml_path, "r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)

    # Check nc
    nc = data_cfg.get("nc")
    if nc != 67:
        raise ValueError(f"Expected nc to be 67, but got {nc} in data.yaml")

    # Get absolute dataset directory path
    dataset_path = Path(data_cfg.get("path", "."))
    
    print("\nDataset Verification:")
    print(f"  Dataset path: {dataset_path.resolve()}")

    split_counts = {}
    for split in ["train", "val", "test"]:
        img_dir = dataset_path / "images" / split
        lbl_dir = dataset_path / "labels" / split

        if not img_dir.exists():
            raise FileNotFoundError(f"Images directory missing: {img_dir}")
        if not lbl_dir.exists():
            raise FileNotFoundError(f"Labels directory missing: {lbl_dir}")

        image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        images = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in image_exts])
        labels = sorted([p for p in lbl_dir.iterdir() if p.suffix.lower() == ".txt"])

        split_counts[split] = len(images)
        print(f"  Split '{split}': Found {len(images)} images and {len(labels)} label files.")

        if len(images) != len(labels):
            raise ValueError(f"Mismatch: split '{split}' has {len(images)} images but {len(labels)} labels.")

        # Check pairing stems
        img_stems = {p.stem for p in images}
        lbl_stems = {p.stem for p in labels}
        mismatches = img_stems.symmetric_difference(lbl_stems)
        if mismatches:
            raise ValueError(f"Mismatch: split '{split}' files do not match 1-to-1. Stems: {list(mismatches)[:5]}")

        # Check label class range [0, 66]
        for lbl_path in labels:
            if lbl_path.stat().st_size == 0:
                continue
            with open(lbl_path, "r", encoding="utf-8") as f:
                for line_idx, line in enumerate(f, 1):
                    stripped = line.strip()
                    if not stripped:
                        continue
                    tokens = stripped.split()
                    if not tokens:
                        continue
                    try:
                        class_id = int(tokens[0])
                    except ValueError:
                        raise ValueError(f"Invalid class ID format at {lbl_path}:{line_idx}")

                    if class_id < 0 or class_id >= nc:
                        raise ValueError(
                            f"Class ID {class_id} is out of bounds [0, {nc-1}] at {lbl_path}:{line_idx}"
                        )

    print("PASS: Dataset validation passed successfully! All annotations are formatted correctly.")
    return split_counts


def run_visual_sanity_check(data_yaml_path: Path, output_dir: Path):
    """
    Draws ground truth boxes on 3 random training images and saves them to reports/experiments/yolo_baseline/sanity_previews/.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(data_yaml_path, "r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)

    dataset_path = Path(data_cfg.get("path"))
    img_dir = dataset_path / "images" / "train"
    lbl_dir = dataset_path / "labels" / "train"

    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in image_exts])
    
    # Filter only images that contain annotations (size > 0 for label file)
    valid_images = []
    for img_path in images:
        lbl_path = lbl_dir / f"{img_path.stem}.txt"
        if lbl_path.exists() and lbl_path.stat().st_size > 0:
            valid_images.append(img_path)

    if not valid_images:
        print("No annotated images found for visual sanity check.")
        return

    random.seed(42)
    sample_imgs = random.sample(valid_images, min(3, len(valid_images)))

    colors = [
        (0, 255, 0),    # Green
        (255, 0, 0),    # Blue
        (0, 0, 255),    # Red
        (0, 255, 255),  # Yellow
        (255, 0, 255)   # Magenta
    ]

    print(f"\nGenerating visual sanity previews in: {output_dir.resolve()}")
    for idx, img_path in enumerate(sample_imgs, start=1):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        lbl_path = lbl_dir / f"{img_path.stem}.txt"
        with open(lbl_path, "r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if not stripped:
                    continue
                tokens = stripped.split()
                if len(tokens) != 5:
                    continue
                
                class_id = int(tokens[0])
                xc, yc, bw, bh = map(float, tokens[1:])

                # Convert to absolute pixel boundaries
                x1 = int((xc - bw / 2) * w)
                y1 = int((yc - bh / 2) * h)
                x2 = int((xc + bw / 2) * w)
                y2 = int((yc + bh / 2) * h)

                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w - 1, x2), min(h - 1, y2)

                color = colors[class_id % len(colors)]
                cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness=3)
                
                label_text = f"Class {class_id}"
                cv2.putText(img, label_text, (x1, max(20, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        preview_path = output_dir / f"sanity_preview_{idx}_{img_path.stem}.jpg"
        cv2.imwrite(str(preview_path), img)
        print(f"  Saved sanity preview: {preview_path.name}")


def main():
    parser = argparse.ArgumentParser(description="Ultralytics YOLO Direct SKU Baseline Training Runner")
    parser.add_argument(
        "--data",
        type=str,
        default="data/processed/yolo_remapped/data.yaml",
        help="Path to dataset data.yaml file"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="Ultralytics YOLO model weight file (e.g. yolov8n.pt, yolov8s.pt)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of epochs to train"
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Image target size"
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Training execution device ('cpu', '0', 'cuda')"
    )
    parser.add_argument(
        "--project",
        type=str,
        default="runs/detect",
        help="Model outputs logging project folder"
    )
    parser.add_argument(
        "--name",
        type=str,
        default="yolo_baseline_smoke",
        help="Name of the training run"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Reproducibility random seed"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of dataloader workers"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early stopping patience epochs count"
    )

    args = parser.parse_args()

    data_yaml = Path(args.data)
    sanity_dir = Path("reports/experiments/yolo_baseline/sanity_previews")

    print("==================================================")
    print("Direct YOLO Baseline - Dataset Verification & training launcher")
    print(f"data.yaml:    {data_yaml.resolve()}")
    print(f"Model weight: {args.model}")
    print(f"Epochs count: {args.epochs}")
    print(f"Image size:   {args.imgsz}")
    print(f"Batch size:   {args.batch}")
    print(f"Device:       {args.device}")
    print(f"Output name:  {args.name}")
    print("==================================================")

    # 1. Validate dataset
    validate_yolo_dataset(data_yaml)

    # 2. Run sanity visuals check
    run_visual_sanity_check(data_yaml, sanity_dir)

    # 3. Load YOLO and execute training
    print(f"\nImporting Ultralytics YOLO to train on device='{args.device}'...")
    from ultralytics import YOLO

    print(f"Initializing YOLO model: {args.model}...")
    model = YOLO(args.model)

    print("Launching training execution loop...")
    abs_project_path = str(Path(args.project).resolve())
    results = model.train(
        data=str(data_yaml.resolve()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=abs_project_path,
        name=args.name,
        seed=args.seed,
        workers=args.workers,
        patience=args.patience
    )

    print("\nYOLO Training Execution Finished Successfully!")
    print(f"Best model weights saved in: {abs_project_path}/{args.name}/weights/best.pt")
    print("==================================================")


if __name__ == "__main__":
    main()
