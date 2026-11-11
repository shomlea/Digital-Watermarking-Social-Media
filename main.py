import cv2
from watermark import embed_watermark, extract_watermark

# --- embed watermark ---
# alpha: embedding strength (0.01 to 0.10). 0.05 is a solid default
watermarked_image, metadata = embed_watermark(
    cover_path="images/dog.jpg",
    watermark_path="logo.png",
    alpha=0.05,
    wavelet="haar"
)

watermarked_image2, metadata2 = embed_watermark(
    cover_path="images/monke.png",
    watermark_path="logo.png",
    alpha=0.05,
    wavelet="haar"
)

# save the watermarked image
cv2.imwrite("output/watermarked_output.png", watermarked_image)
cv2.imwrite("output/watermarked_output2.png", watermarked_image2)

# --- extract watermark ---
# in a non-blind scheme, the extraction function requires:
# the watermarked (or attacked) image
# the metadata dictionary saved during embedding (U_w, Vt_w, S_c, alpha, wm_shape)
recovered_watermark = extract_watermark(watermarked_image, metadata, wavelet="haar")
recovered_watermark2 = extract_watermark(watermarked_image2, metadata2, wavelet="haar")


no_watermark_img = cv2.imread(
    "images/dog.jpg",
    cv2.IMREAD_COLOR
)

no_watermark = extract_watermark(
    no_watermark_img,
    metadata
)


cv2.imwrite(
    "output/no_watermark.png",
    no_watermark
)

# save the recovered watermark
cv2.imwrite("output/recovered_watermark.png", recovered_watermark)
cv2.imwrite("output/recovered_watermark2.png", recovered_watermark2)

