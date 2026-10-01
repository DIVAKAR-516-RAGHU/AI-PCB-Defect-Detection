from pathlib import Path
from PIL import Image

ROOT = Path("dataset/YOLO")
NUM_CLASSES = 8

errors = []
checked = 0

for split in ["train", "val", "test"]:
    image_dir = ROOT / "images" / split
    label_dir = ROOT / "labels" / split

    images = list(image_dir.glob("*"))
    labels = {x.stem for x in label_dir.glob("*.txt")}

    for image_path in images:
        if image_path.suffix.lower() not in [".jpg", ".jpeg", ".png", ".bmp"]:
            continue

        checked += 1

        if image_path.stem not in labels:
            errors.append(f"Missing label: {split}/{image_path.name}")
            continue

        try:
            with Image.open(image_path) as img:
                img.verify()
        except Exception as e:
            errors.append(f"Bad image: {split}/{image_path.name} -> {e}")

        label_path = label_dir / f"{image_path.stem}.txt"

        try:
            for line_no, line in enumerate(
                label_path.read_text().splitlines(), 1
            ):
                values = line.split()

                if len(values) != 5:
                    errors.append(
                        f"Bad format: {label_path} line {line_no}"
                    )
                    continue

                class_id = int(values[0])
                coords = [float(x) for x in values[1:]]

                if not 0 <= class_id < NUM_CLASSES:
                    errors.append(
                        f"Invalid class ID: {label_path} line {line_no}"
                    )

                if not all(0 <= x <= 1 for x in coords):
                    errors.append(
                        f"Invalid coordinates: {label_path} line {line_no}"
                    )

        except Exception as e:
            errors.append(f"Bad label: {label_path} -> {e}")

print("=" * 50)
print("PCB DATASET VALIDATION")
print("=" * 50)
print(f"Images checked : {checked}")
print(f"Errors found   : {len(errors)}")

if errors:
    print("\nFIRST 20 ERRORS:")
    for error in errors[:20]:
        print("-", error)
else:
    print("\nALL CHECKS PASSED!")
    print("Dataset is ready for YOLO training.")