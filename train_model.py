import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from data_preprocessing import load_train_data

def build_model(input_shape=(128, 128, 3)):
    """
    Construit un modèle CNN simple.
    """
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')  # Classification binaire
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    # Chemin des données
    train_dir = "dataset/real_and_fake_face"

    # Charger les données
    train_data, val_data = load_train_data(train_dir)

    # Construire le modèle
    model = build_model()

    # Entraîner le modèle
    model.fit(train_data, validation_data=val_data, epochs=10)

    # Sauvegarder le modèle
    model.save('models/real_fake_model.h5')
