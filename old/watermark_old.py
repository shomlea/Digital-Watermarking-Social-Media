import cv2
import numpy as np
import pywt
from scipy.fftpack import dct, idct


def dct2(a):
    return dct(dct(a.T, norm='ortho').T, norm='ortho')


def idct2(a):
    return idct(idct(a.T, norm='ortho').T, norm='ortho')


def embed_watermark(cover_path, watermark_path, alpha=0.05, wavelet='haar'):
    cover_bgr = cv2.imread(cover_path, cv2.IMREAD_COLOR)

    # convert to YCrCb and split channels
    cover_ycrcb = cv2.cvtColor(cover_bgr, cv2.COLOR_BGR2YCrCb)
    Y, Cr, Cb = cv2.split(cover_ycrcb)
    Y = Y.astype(np.float64) / 255.0

    # read watermark as grayscale and resize to match Y's LL subband
    h, w = Y.shape
    wm_target_shape = (h // 2, w // 2)
    watermark = cv2.imread(watermark_path, cv2.IMREAD_GRAYSCALE).astype(np.float64) / 255.0
    watermark = cv2.resize(watermark, (wm_target_shape[1], wm_target_shape[0]))

    # DWT -> DCT -> SVD on the Y channel
    coeffs = pywt.dwt2(Y, wavelet)
    LL, (LH, HL, HH) = coeffs
    dct_LL = dct2(LL)

    U_c, S_c, Vt_c = np.linalg.svd(dct_LL, full_matrices=True)
    U_w, S_w, Vt_w = np.linalg.svd(watermark, full_matrices=True)

    # embed watermark
    S_mod = S_c + alpha * S_w

    Sigma_mod = np.zeros_like(dct_LL)
    np.fill_diagonal(Sigma_mod, S_mod)
    dct_LL_mod = np.dot(U_c, np.dot(Sigma_mod, Vt_c))

    LL_mod = idct2(dct_LL_mod)
    Y_watermarked = pywt.idwt2((LL_mod, (LH, HL, HH)), wavelet)
    Y_watermarked = np.clip(Y_watermarked, 0.0, 1.0)
    Y_mod_uint8 = (Y_watermarked * 255).astype(np.uint8)

    # merge back with untouched Cr and Cb channels, convert to BGR
    watermarked_ycrcb = cv2.merge([Y_mod_uint8, Cr, Cb])
    watermarked_bgr = cv2.cvtColor(watermarked_ycrcb, cv2.COLOR_YCrCb2BGR)

    metadata = {
        'U_w': U_w,
        'Vt_w': Vt_w,
        'S_c': S_c,
        'alpha': alpha,
        'wm_shape': watermark.shape
    }

    return watermarked_bgr, metadata


def extract_watermark(watermarked_bgr_img, metadata, wavelet='haar'):
    # convert watermarked image to YCrCb and take Y channel
    ycrcb = cv2.cvtColor(watermarked_bgr_img, cv2.COLOR_BGR2YCrCb)
    Y, _, _ = cv2.split(ycrcb)
    Y = Y.astype(np.float64) / 255.0

    # DWT -> DCT -> SVD on Y channel
    LL_wm, _ = pywt.dwt2(Y, wavelet)
    dct_LL_wm = dct2(LL_wm)
    _, S_wm, _ = np.linalg.svd(dct_LL_wm, full_matrices=True)

    # extract singular values
    S_extracted = (S_wm - metadata['S_c']) / metadata['alpha']

    # reconstruct watermark image
    Sigma_extracted = np.zeros(metadata['wm_shape'])
    np.fill_diagonal(Sigma_extracted, S_extracted)
    extracted_wm = np.dot(metadata['U_w'], np.dot(Sigma_extracted, metadata['Vt_w']))
    extracted_wm = np.clip(extracted_wm, 0.0, 1.0)

    return (extracted_wm * 255).astype(np.uint8)