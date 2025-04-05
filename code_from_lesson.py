import matplotlib.pyplot as plt
from skimage.measure import regionprops, label
from skimage.morphology import binary_dilation
from skimage.filters import threshold_otsu
import numpy as np


image = plt.imread("balls.png")

if image.ndim == 3:
    image = image.mean(axis=2)

thresh = threshold_otsu(image)
binary_image = image > thresh

dilated_image = binary_dilation(binary_image)

labeled = label(dilated_image)

regions = regionprops(labeled)
number_of_balls = len(regions)

print(f"Количество найденных шариков: {number_of_balls}")

plt.imshow(dilated_image, cmap='gray')
plt.title('Binary Image with Dilated Balls')
plt.axis('off')
plt.show()
