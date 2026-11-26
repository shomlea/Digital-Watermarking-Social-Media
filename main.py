import glob
import os

import cv2

from src.attacks import jpeg, rescale
from src.watermark import METHODS, detect_watermark, embed_watermark, save_metadata

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(BASE_DIR, "images")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

image_paths = sorted(glob.glob(os.path.join(IMAGES_DIR, "commons_*.png")))
print(f"found {len(image_paths)} images")

for path in image_paths:
    name = os.path.splitext(os.path.basename(path))[0]
    cover = cv2.imread(path, cv2.IMREAD_COLOR)
    print(f"\n{name}")

    for method in METHODS:
        watermarked_image, metadata = embed_watermark(
            cover_path=path,
            method=method,
            n_bits=64,
            delta=0.2,
            seed=42
        )

        os.makedirs(os.path.join(OUTPUT_DIR, method), exist_ok=True)
        cv2.imwrite(os.path.join(OUTPUT_DIR, method, name + ".png"), watermarked_image)
        save_metadata(metadata, os.path.join(OUTPUT_DIR, method, name + ".json"))

        tests = [
            ("watermarked", watermarked_image),
            ("JPEG 75", jpeg(watermarked_image, 75)),
            ("50% resize", rescale(watermarked_image, 0.5)),
            ("original", cover)
        ]
        for test_name, img in tests:
            result = detect_watermark(img, metadata)
            print(f"  {method:12s} {test_name:12s} BER={result['ber']:.3f}  "
                  f"p={result['p_value']:.1e}  detected={result['detected']}")