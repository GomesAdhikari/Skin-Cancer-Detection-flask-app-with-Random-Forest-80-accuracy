import os
import time
import cv2  # Import OpenCV for image resizing
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import numpy as np
import joblib
from skimage import io, color, filters, feature
from skimage.filters import gabor
import threading

# Initialize Flask app
app = Flask(__name__)

# Set the folder for uploads
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the pre-trained Random Forest model
model = joblib.load('random_forest_model.pkl')

# Feature extraction functions
def extract_lbp_features(image, radius=1, n_points=8):
    lbp = feature.local_binary_pattern(image, n_points, radius, method='uniform')
    lbp_hist, _ = np.histogram(lbp, bins=np.arange(0, n_points + 3), density=True)
    return lbp_hist

def extract_gabor_features(image, frequencies):
    gabor_features = []
    for frequency in frequencies:
        real, imag = gabor(image, frequency=frequency)
        gabor_features.append(np.mean(real))
        gabor_features.append(np.mean(imag))
    return np.hstack(gabor_features)

def extract_all_features(gray_image):
    threshold_value = filters.threshold_otsu(gray_image)
    binary_image = gray_image > threshold_value
    edges = feature.canny(gray_image, sigma=1)

    original_lbp_features = extract_lbp_features(gray_image)
    threshold_lbp_features = extract_lbp_features(binary_image.astype(float))
    edge_lbp_features = extract_lbp_features(edges.astype(float))
    gabor_features = extract_gabor_features(gray_image, frequencies=[0.1, 0.2, 0.3])

    combined_feature_vector = np.hstack(
        (original_lbp_features, threshold_lbp_features, edge_lbp_features, gabor_features))
    return combined_feature_vector

# Home page - Welcome Page
@app.route('/')
def welcome():
    return render_template('welcome.html')

# Image upload page
@app.route('/upload')
def index():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Load and resize the image
        image = io.imread(file_path)
        image_resized = cv2.resize(image, (128, 128))  # Resize to 128x128
        gray_image = color.rgb2gray(image_resized)
        features = extract_all_features(gray_image)

        prediction = model.predict([features])
        probability = model.predict_proba([features])

        cancer_type = 'Malignant' if prediction == 1 else 'Benign'
        probability_percent = np.max(probability) * 100

        # Schedule deletion after rendering
        def delete_file(path):
            time.sleep(5)
            if os.path.exists(path):
                os.remove(path)

        threading.Thread(target=delete_file, args=(file_path,)).start()

        return render_template(
            'index.html',
            filename=filename,
            prediction=cancer_type,
            probability_benign=probability_percent if prediction == 0 else 100 - probability_percent,
            probability_malignant=100 - (probability_percent if prediction == 0 else 0)
        )

# To display uploaded images
@app.route('/uploads/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

if __name__ == '__main__':
    app.run(debug=True)
