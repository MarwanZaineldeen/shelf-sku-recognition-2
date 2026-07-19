import argparse
import random
import sys
from pathlib import Path

# Add root folder to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))


def main():
    parser = argparse.ArgumentParser(description="Ultralytics YOLO Inference Wrapper for Test Images")
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to YOLO best.pt weights file"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="data/processed/yolo_remapped/images/test",
        help="Path to test images folder or a single image path"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports/experiments/yolo_baseline/inference_previews",
        help="Path to save annotated prediction previews"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=20,
        help="Number of random test images to run inference on"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for predictions"
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size"
    )

    args = parser.parse_args()

    weights_path = Path(args.weights)
    source_path = Path(args.source)
    output_dir = Path(args.output_dir)

    print("==================================================")
    print("YOLO Inference Previews Runner")
    print(f"Weights:     {weights_path.resolve()}")
    print(f"Source:      {source_path.resolve()}")
    print(f"Output Dir:  {output_dir.resolve()}")
    print(f"Samples count:{args.num_samples}")
    print(f"Confidence:  {args.conf}")
    print("==================================================")

    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found at: {weights_path}")

    # Import YOLO
    from ultralytics import YOLO

    model = YOLO(args.weights)

    # Gather test images
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    if source_path.is_file():
        test_images = [source_path]
    elif source_path.is_dir():
        test_images = sorted([p for p in source_path.iterdir() if p.suffix.lower() in image_exts])
    else:
        raise FileNotFoundError(f"Source path not found: {source_path}")

    if not test_images:
        print(f"No test images found in source path: {source_path}")
        return

    # Sample images
    random.seed(42)
    sample_imgs = random.sample(test_images, min(args.num_samples, len(test_images)))

    # Run inference and save outputs
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nRunning predictions on {len(sample_imgs)} sampled images...")
    for idx, img_path in enumerate(sample_imgs, start=1):
        print(f"  Predicting [{idx}/{len(sample_imgs)}]: {img_path.name}")
        results = model.predict(
            source=str(img_path),
            imgsz=args.imgsz,
            conf=args.conf,
            device="cpu"
        )
        
        # Save results (ultralytics saves them via result.save())
        for result in results:
            save_path = output_dir / f"pred_{img_path.name}"
            result.save(filename=str(save_path))
            
    print(f"\nInference completed. Results saved in: {output_dir.resolve()}")
    print("==================================================")


if __name__ == "__main__":
    main()
