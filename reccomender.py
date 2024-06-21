from flask import Flask, request, render_template, redirect, url_for, send_from_directory
import cv2
import numpy as np
import os
import csv

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
WARDROBE_FOLDER = 'wardrobe_images'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['WARDROBE_FOLDER'] = WARDROBE_FOLDER
FEATURES_FILE = 'wardrobe_features.csv'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        similar_items = find_similar_items(filepath)
        return render_template('result.html', similar_items=similar_items)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/wardrobe_images/<filename>')
def wardrobe_file(filename):
    return send_from_directory(app.config['WARDROBE_FOLDER'], filename)

def extract_features(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    hist = cv2.calcHist([image_rgb], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

def load_wardrobe_features():
    wardrobe_features = []
    with open(FEATURES_FILE, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            filename = row[0]
            features = np.array([float(x) for x in row[1:]])
            wardrobe_features.append((filename, features))
    return wardrobe_features

def find_similar_items(image_path):
    uploaded_features = extract_features(image_path)
    wardrobe_features = load_wardrobe_features()
    similarities = [(filename, np.linalg.norm(uploaded_features - features)) for filename, features in wardrobe_features]
    similarities.sort(key=lambda x: x[1])
    return [item[0] for item in similarities[:3]]  # Return top 3 similar items

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)
