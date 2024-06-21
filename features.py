import cv2
import numpy as np
import os
import csv

# Directory containing wardrobe images
WARDROBE_DIR = 'wardrobe_images'
FEATURES_FILE = 'wardrobe_features.csv'

def extract_features(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    hist = cv2.calcHist([image_rgb], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

def store_features():
    with open(FEATURES_FILE, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['filename'] + [f'feature_{i}' for i in range(512)])
        for filename in os.listdir(WARDROBE_DIR):
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(WARDROBE_DIR, filename)
                features = extract_features(image_path)
                writer.writerow([filename] + features.tolist())

if __name__ == '__main__':
    store_features()
