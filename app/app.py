import os
import shutil
from flask import Flask, render_template, jsonify, request
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from .src.data_loader import load_dataset
from .src.model import train_model, evaluate_model, create_model
import webbrowser

app = Flask(__name__)

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), os.path.join('models', 'model.h5'))
CLASS_NAMES = ["fake", "real"]
UPLOAD_FOLDER = "./uploads"
METRICS_FOLDER = "./metrics"
os.makedirs(METRICS_FOLDER, exist_ok=True)
#define model variable
model = ''
# Charge the data set
train_data, test_data = load_dataset(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'dataset'), img_size=(600, 600))

if os.path.exists(MODEL_PATH):
    # Use the model if already existing
    print(f"Existing model found in {MODEL_PATH}. Launching...")
    model = load_model(MODEL_PATH)
else:
    # Create the model
    model = create_model(input_shape=(600, 600, 3))

    # Train the model
    history = train_model(model, train_data, epochs=10)

    # Evaluate the model
    evaluate_model(model, test_data, history)

def predict_image(image_path, img_size=(600, 600)):
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
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    if os.path.exists(file_path):
        predicted_class, confidence = predict_image(file_path)
        print(f"predicted class : {predicted_class} with confidence {confidence:.2f}")
        return jsonify({"message": f"File {file.filename} save with success. predicted class : {predicted_class} with confidence {confidence:.2f}."}), 200
    else:
        print("The specified directory is not valid or the file does not exist.")
        return jsonify({"message": f"File {file.filename} does not exist."}), 400

@app.route('/evaluate', methods=['GET'])
def evaluate_model_api():
    if not os.path.exists(MODEL_PATH):
        return jsonify({"error": "Model doesn't exist in the directory."}), 400

    # Separe labels
    X_test = []
    y_test = []

    for images, labels in test_data:
        X_test.append(images.numpy())
        y_test.append(labels.numpy())

    X_test = np.concatenate(X_test, axis=0)
    y_test = np.concatenate(y_test, axis=0)

    # Perform prediction
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Metrics calculation
    report = classification_report(y_test, y_pred, target_names=CLASS_NAMES, output_dict=True)
    confusion = confusion_matrix(y_test, y_pred)

    # Save metrics into a JSON file
    report_path = os.path.join(METRICS_FOLDER, "classification_report.json")
    with open(report_path, "w") as f:
        import json
        json.dump(report, f, indent=4)

    print("Classification report saved.")

    # Generate confusion matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel("Prediction")
    plt.ylabel("Ground truth")
    plt.title("Confusion matrix")
    confusion_path = os.path.join(METRICS_FOLDER, "confusion_matrix.png")
    plt.savefig(confusion_path)
    plt.close()

    print("Confusion matrix generated and saved.")

    # Generate ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_probs[:, 1])  # "real" class prob
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="blue", lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    roc_curve_path = os.path.join(METRICS_FOLDER, "roc_curve.png")
    plt.savefig(roc_curve_path)
    plt.close()

    print("ROC curve generated and saved.")

    return jsonify({
        "message": "Model evaluation completed successfully.",
        "classification_report": report_path,
        "confusion_matrix": confusion_path,
        "roc_curve": roc_curve_path
    }), 200

@app.route('/delete', methods=['DELETE'])
def delete_directory():
    try:
        if os.path.exists(UPLOAD_FOLDER):
            # Delete the directory and its content
            shutil.rmtree(UPLOAD_FOLDER)
            return jsonify({"message": "Folder delete with success."}), 200
        else:
            return jsonify({"message": "The folder doesn't exist."}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def main():
    url = f"http://localhost:5000"
    webbrowser.open(url)
    app.run(debug=False, port=5000)