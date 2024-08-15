import cv2
import numpy as np
import pandas as pd
import os
from skimage.feature import local_binary_pattern

def extract_lbp(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    radius = 1
    n_points = 8 * radius
    lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)  # Normalize the histogram
    return hist

def process_image(image_path):
    image = cv2.imread(image_path)
    lbp_hist = extract_lbp(image)
    filename = os.path.basename(image_path)
    lbp_hist_str = ','.join(map(str, lbp_hist))  # Convert histogram to a comma-separated string
    return [filename, lbp_hist_str]

folder_path = "DONE"
data = []

for filename in os.listdir(folder_path):
    if filename.endswith(('.jpg', '.jpeg', '.png', '.JPG')):
        image_path = os.path.join(folder_path, filename)
        features = process_image(image_path)
        data.append(features)

# Convert the list of lists into a DataFrame
df = pd.DataFrame(data, columns=["Filename", "LBP Histogram Value"])

# Save the DataFrame to a CSV file
csv_file_path = "lbpValueJerawat.csv"
df.to_csv(csv_file_path, index=False)

print(f"Data has been saved to {csv_file_path}")
