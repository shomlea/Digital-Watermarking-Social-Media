import cv2
import numpy as np

# traditional benchmark attacks
# every attack takes a BGR uint8 image and returns one
# rescale and crop return a smaller image, like a platform would


def jpeg(img, quality):
    _, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def webp(img, quality):
    _, buf = cv2.imencode('.webp', img, [cv2.IMWRITE_WEBP_QUALITY, int(quality)])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def rescale(img, scale):
    h, w = img.shape[:2]
    return cv2.resize(img, (max(1, round(w * scale)), max(1, round(h * scale))),
                      interpolation=cv2.INTER_AREA)


def gaussian_noise(img, sigma, seed=0):
    noise = np.random.default_rng(seed).normal(0.0, sigma, img.shape)
    return np.clip(img + noise, 0, 255).astype(np.uint8)


def gaussian_blur(img, sigma):
    return cv2.GaussianBlur(img, (0, 0), sigma)


def median(img, k):
    return cv2.medianBlur(img, int(k))


def sharpen(img, amount):
    blur = cv2.GaussianBlur(img, (0, 0), 1.0)
    return cv2.addWeighted(img, 1 + amount, blur, -amount, 0)


def brightness(img, offset):
    return np.clip(img.astype(np.int16) + int(offset), 0, 255).astype(np.uint8)


def contrast(img, factor):
    return np.clip((img.astype(np.float64) - 128) * factor + 128, 0, 255).astype(np.uint8)


# keep the central part of the image, `keep` = fraction of each side
def center_crop(img, keep):
    h, w = img.shape[:2]
    ch, cw = round(h * keep), round(w * keep)
    y, x = (h - ch) // 2, (w - cw) // 2
    return img[y:y + ch, x:x + cw]


def rotate(img, degrees):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), degrees, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)


# (family, strength, function) - strength is in the family's own unit
# (JPEG quality, scale factor, sigma, ...) so every family gives a robustness curve
ATTACKS = [('none', 0, lambda x: x)]
ATTACKS += [('jpeg', q, lambda x, q=q: jpeg(x, q)) for q in (95, 90, 80, 70, 60, 50, 30)]
ATTACKS += [('webp', q, lambda x, q=q: webp(x, q)) for q in (90, 75, 50)]
ATTACKS += [('rescale', s, lambda x, s=s: rescale(x, s)) for s in (0.9, 0.75, 0.5, 0.25)]
ATTACKS += [('noise', s, lambda x, s=s: gaussian_noise(x, s)) for s in (2, 5, 10, 20)]
ATTACKS += [('blur', s, lambda x, s=s: gaussian_blur(x, s)) for s in (0.5, 1.0, 2.0)]
ATTACKS += [('median', k, lambda x, k=k: median(x, k)) for k in (3, 5)]
ATTACKS += [('sharpen', a, lambda x, a=a: sharpen(x, a)) for a in (0.5, 1.0)]
ATTACKS += [('brightness', b, lambda x, b=b: brightness(x, b)) for b in (-30, 30)]
ATTACKS += [('contrast', c, lambda x, c=c: contrast(x, c)) for c in (0.7, 1.3)]
ATTACKS += [('crop', k, lambda x, k=k: center_crop(x, k)) for k in (0.95, 0.8)]
ATTACKS += [('rotate', d, lambda x, d=d: rotate(x, d)) for d in (1, 5)]
