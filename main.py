import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from src.data_loader import load_dataset
from src.model import train_model, evaluate_model, create_model

MODEL_PATH = "models/model.h5"
CLASS_NAMES = ["fake", "real"]

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

if __name__ == "__main__":
    if os.path.exists(MODEL_PATH):
        # Charger le model si existant
        print(f"Modèle existant trouvé à {MODEL_PATH}. Chargement en cours...")
        model = load_model(MODEL_PATH)
    else:
        # Charger les données
        train_data, test_data = load_dataset("dataset", img_size=(600, 600))

        # Créer le modèle
        model = create_model(input_shape=(600, 600, 3))

        # Entraîner le modèle
        history = train_model(model, train_data, epochs=10)

        # Évaluer le modèle
        evaluate_model(model, test_data, history)

    # Exemple de prédiction sur une nouvelle image
    image_path = input("Entrez le chemin de l'image à prédire : ")
    if os.path.exists(image_path):
        predicted_class, confidence = predict_image(model, image_path)
        print(f"Classe prédite : {predicted_class} avec une confiance de {confidence:.2f}")
    else:
        print("Le chemin spécifié de l'image est invalide.")