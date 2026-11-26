import cv2
import numpy as np

# create a 256x256 white canvas
wm = np.full((256, 256), 255, dtype=np.uint8)

# draw bold black text and a border
cv2.putText(wm, "DOG", (25, 145), cv2.FONT_HERSHEY_DUPLEX, 1.6, 0, 4)
cv2.rectangle(wm, (10, 10), (245, 245), 0, 4)

# save as logo.png
cv2.imwrite("logo.png", wm)
print("Saved clean 256x256 logo.png")