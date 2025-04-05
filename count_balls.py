import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import regionprops, label
from pathlib import Path

def neighbours4(y, x):
    return (y-1, x), (y+1, x), (y, x-1), (y, x+1)


def count_balls(region):
    shape = region.image.shape
    new_image = np.zeros((shape[0] + 2, shape[1] + 2), dtype=bool)
    new_image[1: -1, 1:-1] = region.image
    new_image = np.logical_not(new_image)
    labeled = label(new_image)
    return np.max(labeled) - 1

def calculate_centroid(region):
    return region.centroid

def recognize(region):
    if np.all(region.image):
        return "Полный объект"
    
    balls = count_balls(region)
    
    return f"Количество касающийся шариков: {balls}"

def fill_and_find_touching(lb, label, y, x):
    lb[y, x] = label
    for ny, nx in neighbours4(y, x):
        if 0 <= ny < lb.shape[0] and 0 <= nx < lb.shape[1]:
            if lb[ny, nx] == -1:  # Если пиксель не помечен
                fill_and_find_touching(lb, label, ny, nx)
            elif lb[ny, nx] != 0 and lb[ny, nx] != label:  
                print(f"Шарики касаются друг друга на пикселях: ({y}, {x}) и ({ny}, {nx})")

def recursive_labeling(image):
    lb = image * -1
    label = 0
    for y in range(lb.shape[0]):
        for x in range(lb.shape[1]):
            if lb[y, x] == -1 and image[y, x] != 0:
                label += 1
                fill_and_find_touching(lb, label, y, x)
    return lb


symbols = plt.imread(Path(__file__).parent / "balls.png")
gray = symbols[:, :, :-1].mean(axis=2)
binary = gray > 0
labeled = label(binary)
regions = regionprops(labeled)

print(f"Количество найденных объектов: {len(regions)}")

result = {}
out_path = Path(__file__).parent / "out_2"
out_path.mkdir(exist_ok=True)

for i, region in enumerate(regions):
    print(f"{i + 1}/{len(regions)}")
    symbol = recognize(region)
    if symbol not in result:
        result[symbol] = 0
    result[symbol] += 1

    plt.cla()
    plt.title(symbol)
    plt.imshow(region.image)
    plt.savefig(out_path / f"{i:03d}.png")

print(result)
