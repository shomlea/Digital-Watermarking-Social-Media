import cv2
from watermark import embed_watermark, extract_watermark

# --- embed watermark ---
# alpha: embedding strength (0.01 to 0.10). 0.05 is a solid default
watermarked_image, metadata = embed_watermark(
    cover_path="dog.jpg",
    watermark_path="logo.png",
    alpha=0.05,
    wavelet="haar"
)

# save the watermarked image
cv2.imwrite("watermarked_output.png", watermarked_image)
print("Watermarked image saved as 'watermarked_output.png'.")

# --- extract watermark ---
# in a non-blind scheme, the extraction function requires:
# the watermarked (or attacked) image
# the metadata dictionary saved during embedding (U_w, Vt_w, S_c, alpha, wm_shape)
recovered_watermark = extract_watermark(watermarked_image, metadata, wavelet="haar")

no_watermark_img = cv2.imread(
    "dog.jpg",
    cv2.IMREAD_COLOR
)

no_watermark = extract_watermark(
    no_watermark_img,
    metadata
)

cv2.imwrite(
    "no_watermark.png",
    no_watermark
)

# save the recovered watermark
cv2.imwrite("recovered_watermark.png", recovered_watermark)
print("Recovered watermark saved as 'recovered_watermark.png'.")