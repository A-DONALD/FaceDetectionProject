import os
import shutil
from flask import Flask, render_template, jsonify, request
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from .src.data_loader import load_dataset
from .src.model import train_model, evaluate_model, create_model
import webbrowser

app = Flask(__name__)

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), os.path.join('models', 'model.h5'))
CLASS_NAMES = ["fake", "real"]
upload_folder = './uploads'

def predict_image(model, image_path, img_size=(600, 600)):
    # Charge the image and preprocess
    img = load_img(image_path, target_size=img_size)
    img_array = img_to_array(img)  # Convert in numpy
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize pixel between 0 and 1

    # Realize the prediction
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])  # Class with the maximum confidence
    confidence = predictions[0][predicted_class]  # Confidence

    return CLASS_NAMES[predicted_class], confidence

if os.path.exists(MODEL_PATH):
    # Use the model if already existing
    print(f"Modèle existant trouvé à {MODEL_PATH}. Chargement en cours...")
    model = load_model(MODEL_PATH)
else:
    # CHarge the data set
    train_data, test_data = load_dataset(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'dataset'), img_size=(600, 600))

    # Create the model
    model = create_model(input_shape=(600, 600, 3))

    # Train the model
    history = train_model(model, train_data, epochs=10)

    # Evaluate the model
    evaluate_model(model, test_data, history)

@app.route('/')
def hello():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file founded"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "File with no name"}), 400

    # Save the file in the temp upload directory
    os.makedirs(upload_folder, exist_ok=True)
    file_path = os.path.join(upload_folder, file.filename)
    file.save(file_path)

    if os.path.exists(file_path):
        predicted_class, confidence = predict_image(model, file_path)
        print(f"predicted class : {predicted_class} with confidence {confidence:.2f}")
        return jsonify({"message": f"File {file.filename} save with success. predicted class : {predicted_class} with confidence {confidence:.2f}."}), 200
    else:
        print("The specified directory is not valid or the file does not exist.")
        return jsonify({"message": f"File {file.filename} does not exist."}), 400
@app.route('/delete', methods=['DELETE'])
def delete_directory():
    try:
        if os.path.exists(upload_folder):
            # Delete the directory and its content
            shutil.rmtree(upload_folder)
            return jsonify({"message": "Folder delete with success."}), 200
        else:
            return jsonify({"message": "The folder doesn't exist."}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def main():
    url = f"http://localhost:5000"
    webbrowser.open(url)
    app.run(debug=False, port=5000)