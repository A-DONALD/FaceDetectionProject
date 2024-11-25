from train_model import build_model
from data_preprocessing import load_train_data, load_test_data
from Evaluate_model import evaluate_model

if __name__ == "__main__":
    # Chemins des données
    train_dir = "dataset/real_and_fake_face"
    test_dir = "dataset/real_and_fake_face_detection/real_and_fake_face"

    # Étape 1 : Charger les données
    train_data, val_data = load_train_data(train_dir)

    # Étape 2 : Construire et entraîner le modèle
    model = build_model()
    model.fit(train_data, validation_data=val_data, epochs=10)
    model.save('models/real_fake_model.h5')

    # Étape 3 : Évaluer le modèle
    evaluate_model(model_path='models/real_fake_model.h5', test_dir=test_dir)
