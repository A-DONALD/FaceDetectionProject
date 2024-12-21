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
    # Charger l'image et la prétraiter
    img = load_img(image_path, target_size=img_size)
    img_array = img_to_array(img)  # Convertir en tableau numpy
    img_array = np.expand_dims(img_array, axis=0)  # Ajouter une dimension batch
    img_array = img_array / 255.0  # Normaliser les pixels entre 0 et 1

    # Faire la prédiction
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])  # Classe avec la probabilité maximale
    confidence = predictions[0][predicted_class]  # Confiance associée à cette classe

    return CLASS_NAMES[predicted_class], confidence

if os.path.exists(MODEL_PATH):
    # Charger le model si existant
    print(f"Modèle existant trouvé à {MODEL_PATH}. Chargement en cours...")
    model = load_model(MODEL_PATH)
else:
    # Charger les données
    train_data, test_data = load_dataset(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'dataset'), img_size=(600, 600))

    # Créer le modèle
    model = create_model(input_shape=(600, 600, 3))

    # Entraîner le modèle
    history = train_model(model, train_data, epochs=10)

    # Évaluer le modèle
    evaluate_model(model, test_data, history)

@app.route('/')
def hello():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "Aucun fichier trouvé"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "Nom de fichier vide"}), 400

    # Sauvegarder le fichier dans un répertoire temporaire
    os.makedirs(upload_folder, exist_ok=True)
    file_path = os.path.join(upload_folder, file.filename)
    file.save(file_path)

    if os.path.exists(file_path):
        predicted_class, confidence = predict_image(model, file_path)
        print(f"Classe prédite : {predicted_class} avec une confiance de {confidence:.2f}")
        return jsonify({"message": f"Fichier {file.filename} sauvegardé avec succès. Classe prédite : {predicted_class} avec une confiance de {confidence:.2f}."}), 200
        # return jsonify(prediction=predicted_class, confidence=confidence)
    else:
        print("Le chemin spécifié de l'image est invalide.")
        return jsonify({"message": f"Fichier {file.filename} absent du dossier upload."}), 400
        # return jsonify(prediction="", confidence="")

@app.route('/delete', methods=['DELETE'])
def delete_directory():
    try:
        if os.path.exists(upload_folder):
            # Supprimer le répertoire et son contenu
            shutil.rmtree(upload_folder)
            return jsonify({"message": "Le répertoire a été supprimé avec succès."}), 200
        else:
            return jsonify({"message": "Le répertoire n'existe pas."}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def main():
    url = f"http://localhost:5000"
    webbrowser.open(url)
    app.run(debug=False, port=5000)