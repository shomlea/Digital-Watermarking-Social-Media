import csv
import io
import json
import os
import random
import re
import time
import urllib.parse
import urllib.request

import ssl

import certifi

from datetime import datetime, timezone

from PIL import Image, ImageOps

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGES_DIR = os.path.join(BASE_DIR, "images")
SOURCES_CSV = os.path.join(IMAGES_DIR, "sources.csv")

N_IMAGES = 9
CATEGORIES = ["Category:Quality_images"]
SEED = 0

# only real photos of reasonable size
MIN_SIDE = 1500
MAX_MEGAPIXELS = 60
MAX_FILE_MB = 30
MIN_ASPECT, MAX_ASPECT = 0.5, 2.0


TARGET_LONG_SIDE = 2048

USER_AGENT = "WatermarkResearch/0.1 (student project; shomleatudor@gmail.com)"

API_URL = "https://commons.wikimedia.org/w/api.php"
SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())


TIME_RANGE = (datetime(2008, 1, 1, tzinfo=timezone.utc), datetime.now(timezone.utc))


def http_get(url):
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=120, context=SSL_CONTEXT) as response:
        return response.read()


def api_query(category, start_time, limit=20):
    params = {
        "action": "query",
        "format": "json",
        "formatversion": "2",
        "generator": "categorymembers",
        "gcmtitle": category,
        "gcmtype": "file",
        "gcmsort": "timestamp",
        "gcmdir": "newer",
        "gcmstart": start_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "gcmlimit": limit,
        "prop": "imageinfo",
        "iiprop": "url|size|mime|extmetadata",
        "iiextmetadatafilter": "LicenseShortName|LicenseUrl|Artist|UsageTerms"
    }
    data = json.loads(http_get(API_URL + "?" + urllib.parse.urlencode(params)))
    return data.get("query", {}).get("pages", [])


def strip_html(text):
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", text or "")).strip()


def candidate_from_page(page):
    if not page.get("imageinfo"):
        return None
    info = page["imageinfo"][0]
    w, h = info.get("width", 0), info.get("height", 0)
    if info.get("mime") != "image/jpeg":
        return None
    if min(w, h) < MIN_SIDE or w * h > MAX_MEGAPIXELS * 1e6:
        return None
    if not MIN_ASPECT <= w / h <= MAX_ASPECT:
        return None
    if info.get("size", 0) > MAX_FILE_MB * 1024 * 1024:
        return None

    meta = info.get("extmetadata", {})
    license_name = meta.get("LicenseShortName", {}).get("value", "")
    if not license_name:
        return None

    return {
        "title": page["title"],
        "url": info["url"],
        "page_url": info.get("descriptionurl", ""),
        "width": w,
        "height": h,
        "license": license_name,
        "license_url": meta.get("LicenseUrl", {}).get("value", ""),
        "author": strip_html(meta.get("Artist", {}).get("value", ""))
    }


def save_image(raw_bytes, path):
    img = Image.open(io.BytesIO(raw_bytes))
    img = ImageOps.exif_transpose(img).convert("RGB")
    if TARGET_LONG_SIDE is not None and max(img.size) > TARGET_LONG_SIDE:
        scale = TARGET_LONG_SIDE / max(img.size)
        img = img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)
    img.save(path)
    return img.size


def load_sources():
    if not os.path.exists(SOURCES_CSV):
        return []
    with open(SOURCES_CSV, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_sources(rows):
    with open(SOURCES_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    os.makedirs(IMAGES_DIR, exist_ok=True)
    rng = random.Random(SEED)

    sources = load_sources()
    seen = {row["title"] for row in sources}
    print(f"{len(sources)} images already downloaded")

    requests_without_new = 0
    while len(sources) < N_IMAGES:
        category = rng.choice(CATEGORIES)
        start = TIME_RANGE[0] + (TIME_RANGE[1] - TIME_RANGE[0]) * rng.random()
        try:
            pages = api_query(category, start)
        except Exception as e:
            print(f"API request failed ({e}), retrying in 10 s")
            time.sleep(10)
            continue

        candidates = [c for c in map(candidate_from_page, pages) if c and c["title"] not in seen]
        rng.shuffle(candidates)

        added = 0
        for c in candidates[:2]:
            if len(sources) >= N_IMAGES:
                break
            name = f"commons_{len(sources) + 1:03d}"
            try:
                saved_w, saved_h = save_image(http_get(c["url"]), os.path.join(IMAGES_DIR, name + ".png"))
            except Exception as e:
                print(f"  skipping {c['title']} ({e})")
                continue

            seen.add(c["title"])
            sources.append({
                "file": name + ".png",
                "title": c["title"],
                "page_url": c["page_url"],
                "author": c["author"],
                "license": c["license"],
                "license_url": c["license_url"],
                "original_width": c["width"],
                "original_height": c["height"],
                "saved_width": saved_w,
                "saved_height": saved_h,
                "category": category
            })
            write_sources(sources)
            added += 1
            print(f"[{len(sources)}/{N_IMAGES}] {c['title']}  ({c['license']})")
            time.sleep(1)  # be polite to the Wikimedia servers

        requests_without_new = 0 if added else requests_without_new + 1
        if requests_without_new >= 50:
            print("50 requests in a row without a usable image, check the category name and filters")
            break

        time.sleep(0.5)


if __name__ == "__main__":
    main()