import json

import cv2
import numpy as np
import pywt
from scipy.fftpack import dct, idct
from scipy.stats import binom

METHODS = ['dwt_dct', 'dwt_dct_svd']

# DCT coefficient used as the carrier by 'dwt_dct'
DCT_COEF = (0, 1)


# 2D DCT over the last two axes, works on one block or on a whole stack of blocks
def dct2(a):
    return dct(dct(a, axis=-1, norm='ortho'), axis=-2, norm='ortho')


def idct2(a):
    return idct(idct(a, axis=-2, norm='ortho'), axis=-1, norm='ortho')


# (h, w) -> (rows, cols, block, block)
def split_blocks(a, block):
    rows, cols = a.shape[0] // block, a.shape[1] // block
    return a.reshape(rows, block, cols, block).swapaxes(1, 2).copy()


# (rows, cols, block, block) -> (h, w)
def merge_blocks(blocks):
    rows, cols, block, _ = blocks.shape
    return blocks.swapaxes(1, 2).reshape(rows * block, cols * block)


def block_layout(n_blocks, metadata):
    # the seed decides which payload bit each block carries and the dither of each block,
    # so extraction can rebuild both from the metadata alone
    if n_blocks < metadata['n_bits']:
        raise ValueError(f"image too small: {n_blocks} blocks for {metadata['n_bits']} bits")
    rng = np.random.default_rng([metadata['seed'], 2])
    bit_of_block = np.empty(n_blocks, dtype=np.int64)
    bit_of_block[rng.permutation(n_blocks)] = np.arange(n_blocks) % metadata['n_bits']
    dither = rng.uniform(0.0, metadata['delta'], n_blocks)
    return bit_of_block, dither


def y_to_dct_blocks(Y, metadata):
    # use the largest top-left region whose LL subband splits exactly into blocks
    # (this also handles odd image sizes), the rest of the image stays untouched
    step = 2 * metadata['block']
    H, W = (Y.shape[0] // step) * step, (Y.shape[1] // step) * step

    # DWT -> blocks of LL -> DCT of every block
    LL, (LH, HL, HH) = pywt.dwt2(Y[:H, :W], metadata['wavelet'], mode='periodization')
    dct_blocks = dct2(split_blocks(LL, metadata['block']))
    return dct_blocks, (LH, HL, HH), H, W


def qim_embed(x, bits, dither, delta):
    # bit 0 lives on k*delta + dither, bit 1 on k*delta + dither + delta/2
    shift = dither + bits * (delta / 2.0)
    return np.round((x - shift) / delta) * delta + shift


def qim_decode(x, dither, delta):
    # where the value sits inside its cell: near 0 -> bit 0, near 0.5 -> bit 1
    pos = np.mod(x - dither, delta) / delta
    return (np.abs(pos - 0.5) < 0.25).astype(np.int8)


def embed_watermark(cover_path, method='dwt_dct_svd', n_bits=64, delta=0.2, seed=0, block=4,
                    wavelet='haar'):
    cover_bgr = cv2.imread(cover_path, cv2.IMREAD_COLOR)
    if cover_bgr is None:
        raise FileNotFoundError(cover_path)

    # the watermark is a random bit string derived from the seed
    payload = np.random.default_rng([seed, 1]).integers(0, 2, n_bits)

    metadata = {
        'method': method,
        'seed': int(seed),
        'n_bits': int(n_bits),
        'payload': payload.tolist(),
        'delta': float(delta),
        'block': int(block),
        'wavelet': wavelet,
        'shape': [int(cover_bgr.shape[0]), int(cover_bgr.shape[1])]
    }

    # convert to YCrCb and split channels
    cover_ycrcb = cv2.cvtColor(cover_bgr, cv2.COLOR_BGR2YCrCb)
    Y, Cr, Cb = cv2.split(cover_ycrcb)
    Y = Y.astype(np.float64) / 255.0

    # DWT -> DCT on the Y channel
    dct_blocks, details, H, W = y_to_dct_blocks(Y, metadata)
    rows, cols = dct_blocks.shape[:2]
    bit_of_block, dither = block_layout(rows * cols, metadata)
    bits = payload[bit_of_block].reshape(rows, cols)
    dither = dither.reshape(rows, cols)

    # embed one bit per block with QIM
    if method == 'dwt_dct':
        carrier = dct_blocks[..., DCT_COEF[0], DCT_COEF[1]]
        dct_blocks[..., DCT_COEF[0], DCT_COEF[1]] = qim_embed(carrier, bits, dither, delta)
    elif method == 'dwt_dct_svd':
        # carrier = largest singular value of each block's DCT matrix
        U_c, S_c, Vt_c = np.linalg.svd(dct_blocks)
        S_c[..., 0] = np.maximum(qim_embed(S_c[..., 0], bits, dither, delta), 0.0)
        dct_blocks = np.matmul(U_c, S_c[..., :, None] * Vt_c)  # U * S * Vt for every block
    else:
        raise ValueError(f'method must be one of {METHODS}')

    # inverse DCT -> inverse DWT
    LL_mod = merge_blocks(idct2(dct_blocks))
    Y_watermarked = Y.copy()
    Y_watermarked[:H, :W] = pywt.idwt2((LL_mod, details), wavelet, mode='periodization')
    Y_watermarked = np.clip(Y_watermarked, 0.0, 1.0)
    Y_mod_uint8 = np.round(Y_watermarked * 255).astype(np.uint8)

    # merge back with untouched Cr and Cb channels, convert to BGR
    watermarked_ycrcb = cv2.merge([Y_mod_uint8, Cr, Cb])
    watermarked_bgr = cv2.cvtColor(watermarked_ycrcb, cv2.COLOR_YCrCb2BGR)

    return watermarked_bgr, metadata


def extract_watermark(watermarked_bgr_img, metadata):
    if watermarked_bgr_img is None:
        raise ValueError('image is None (wrong path?)')
    if watermarked_bgr_img.ndim == 2:
        watermarked_bgr_img = cv2.cvtColor(watermarked_bgr_img, cv2.COLOR_GRAY2BGR)

    # platforms resize images, so scale back to the size used at embedding
    # (crops and aspect-ratio changes are NOT undone)
    h, w = metadata['shape']
    if watermarked_bgr_img.shape[:2] != (h, w):
        interp = cv2.INTER_CUBIC if watermarked_bgr_img.shape[0] < h else cv2.INTER_AREA
        watermarked_bgr_img = cv2.resize(watermarked_bgr_img, (w, h), interpolation=interp)

    # convert to YCrCb and take Y channel
    ycrcb = cv2.cvtColor(watermarked_bgr_img, cv2.COLOR_BGR2YCrCb)
    Y, _, _ = cv2.split(ycrcb)
    Y = Y.astype(np.float64) / 255.0

    # DWT -> DCT on the Y channel, read the carrier of every block
    dct_blocks, _, _, _ = y_to_dct_blocks(Y, metadata)
    if metadata['method'] == 'dwt_dct':
        carrier = dct_blocks[..., DCT_COEF[0], DCT_COEF[1]]
    else:
        _, S_wm, _ = np.linalg.svd(dct_blocks)
        carrier = S_wm[..., 0]

    # one vote per block, then majority vote per payload bit
    bit_of_block, dither = block_layout(carrier.size, metadata)
    votes = qim_decode(carrier.ravel(), dither, metadata['delta'])
    n_bits = metadata['n_bits']
    ones = np.bincount(bit_of_block, weights=votes, minlength=n_bits)
    total = np.bincount(bit_of_block, minlength=n_bits)
    bits = (ones / total > 0.5).astype(np.uint8)

    # raw BER = share of wrong blocks before voting (0.5 = nothing left)
    # unlike the decoded BER it doesn't get stuck at 0, so it shows how much damage was done
    payload = np.array(metadata['payload'])
    raw_ber = float(np.mean(votes != payload[bit_of_block]))

    return bits, raw_ber


def detect_watermark(img_bgr, metadata, fpr=1e-3):
    # on an image without this watermark the extracted bits are coin flips,
    # so the number of matching bits k follows Binomial(n_bits, 0.5)
    # p_value = P(at least k matches by chance), detected if p_value < fpr
    bits, raw_ber = extract_watermark(img_bgr, metadata)
    payload = np.array(metadata['payload'])
    n_bits = len(payload)
    matches = int(np.sum(bits == payload))
    p_value = float(binom.sf(matches - 1, n_bits, 0.5))

    return {
        'ber': 1 - matches / n_bits,
        'raw_ber': raw_ber,
        'matches': matches,
        'n_bits': n_bits,
        'p_value': p_value,
        'detected': p_value < fpr
    }


def save_metadata(metadata, path):
    with open(path, 'w') as f:
        json.dump(metadata, f, indent=2)


def load_metadata(path):
    with open(path) as f:
        return json.load(f)
