import tensorflow as tf
from data_preprocessing import load_test_data
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import json
import os

# Chemin du dossier "results"
results_dir = "results"

# Vérifier si le dossier existe, sinon le créer
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
    print(f"Dossier '{results_dir}' créé avec succès.")
else:
    print(f"Dossier '{results_dir}' existe déjà.")

def evaluate_model(model_path, test_dir):
    """
    Charge un modèle sauvegardé et l'évalue sur les données de test.
    """
    # Charger le modèle
    model = tf.keras.models.load_model(model_path)

    # Charger les données de test
    test_data = load_test_data(test_dir)

    # Faire des prédictions
    y_pred = (model.predict(test_data) > 0.5).astype("int32")
    y_true = test_data.classes

    # Calculer les métriques
    print(classification_report(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred)
    print(cm)

    # Sauvegarder les résultats
    with open('results/metrics.json', 'w') as f:
        json.dump(classification_report(y_true, y_pred, output_dict=True), f)

    # Sauvegarder la matrice de confusion
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, cmap='Blues', interpolation='nearest')
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.savefig('results/confusion_matrix.png')

if __name__ == "__main__":
    # Chemin du modèle et des données de test
    model_path = 'models/real_fake_model.h5'
    test_dir = "dataset/real_and_fake_face_detection/real_and_fake_face"

    # Évaluer le modèle
    evaluate_model(model_path, test_dir)
