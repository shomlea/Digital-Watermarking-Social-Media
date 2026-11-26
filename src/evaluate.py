import csv
import os
import zlib

import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from attacks import ATTACKS
from watermark import METHODS, detect_watermark, embed_watermark, save_metadata

IMAGES_DIR = "images"
OUTPUT_DIR = "output"
N_BITS = 64
# delta: embedding strength (0.1 to 0.3). 0.2 gives ~42 dB PSNR, higher = more robust but more visible
DELTA = 0.2
# target false-positive rate of the detector (0.1%)
FPR = 1e-3
BASE_SEED = 0


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


image_files = sorted(
    f for f in os.listdir(IMAGES_DIR)
    if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"))
)

raw_rows = []
quality_rows = []

for i, filename in enumerate(image_files, 1):
    path = os.path.join(IMAGES_DIR, filename)
    name = os.path.splitext(filename)[0]
    cover = cv2.imread(path, cv2.IMREAD_COLOR)
    if cover is None:
        print(f"skipping unreadable {path}")
        continue

    for method in METHODS:
        seed = zlib.crc32(f"{BASE_SEED}/{name}/{method}".encode())
        watermarked, metadata = embed_watermark(
            cover_path=path,
            method=method,
            n_bits=N_BITS,
            delta=DELTA,
            seed=seed
        )

        os.makedirs(os.path.join(OUTPUT_DIR, "watermarked", method), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, "metadata", method), exist_ok=True)
        cv2.imwrite(os.path.join(OUTPUT_DIR, "watermarked", method, name + ".png"), watermarked)
        save_metadata(metadata, os.path.join(OUTPUT_DIR, "metadata", method, name + ".json"))

        quality_rows.append({
            "image": name,
            "method": method,
            "height": cover.shape[0],
            "width": cover.shape[1],
            "delta": DELTA,
            "psnr": peak_signal_noise_ratio(cover, watermarked),
            "ssim": structural_similarity(cover, watermarked, channel_axis=2)
        })

        for family, strength, attack in ATTACKS:
            for role, img in [("watermarked", watermarked), ("original", cover)]:
                result = detect_watermark(attack(img), metadata, fpr=FPR)
                raw_rows.append({
                    "image": name,
                    "method": method,
                    "family": family,
                    "strength": strength,
                    "role": role,
                    "ber": result["ber"],
                    "raw_ber": result["raw_ber"],
                    "matches": result["matches"],
                    "n_bits": result["n_bits"],
                    "p_value": result["p_value"],
                    "detected": int(result["detected"])
                })

    print(f"[{i}/{len(image_files)}] {filename}")

if not raw_rows:
    raise SystemExit(f"no images found in {IMAGES_DIR}")

groups = {}
for r in raw_rows:
    groups.setdefault((r["method"], r["family"], r["strength"]), []).append(r)

summary_rows = []
for (method, family, strength), rows in groups.items():
    wm = [r for r in rows if r["role"] == "watermarked"]
    orig = [r for r in rows if r["role"] == "original"]
    summary_rows.append({
        "method": method,
        "family": family,
        "strength": strength,
        "n_images": len(wm),
        "mean_ber": np.mean([r["ber"] for r in wm]),
        "mean_raw_ber": np.mean([r["raw_ber"] for r in wm]),
        "tpr": np.mean([r["detected"] for r in wm]),
        "fpr": np.mean([r["detected"] for r in orig]),
        "mean_ber_original": np.mean([r["ber"] for r in orig])
    })

os.makedirs(OUTPUT_DIR, exist_ok=True)
write_csv(os.path.join(OUTPUT_DIR, "benchmark_raw.csv"), raw_rows)
write_csv(os.path.join(OUTPUT_DIR, "benchmark_summary.csv"), summary_rows)
write_csv(os.path.join(OUTPUT_DIR, "quality.csv"), quality_rows)

print("\nimperceptibility")
for method in METHODS:
    q = [r for r in quality_rows if r["method"] == method]
    print(f"  {method:12s} PSNR {np.mean([r['psnr'] for r in q]):6.2f} dB   "
          f"SSIM {np.mean([r['ssim'] for r in q]):.4f}")

print(f"\n{'method':12s} {'attack':18s} {'BER':>6s} {'rawBER':>7s} {'TPR':>5s} {'FPR':>5s}")
for s in summary_rows:
    attack_name = f"{s['family']} {s['strength']}"
    print(f"{s['method']:12s} {attack_name:18s} {s['mean_ber']:6.3f} {s['mean_raw_ber']:7.3f} "
          f"{s['tpr']:5.2f} {s['fpr']:5.2f}")
