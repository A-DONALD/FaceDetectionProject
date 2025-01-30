import os
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import layers, models

def train_model(model, train_data, epochs=10, save_path="models/model.h5"):
    history = model.fit(train_data, epochs=epochs)
    # Save the model
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    return history

def create_model(input_shape):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(2, activation='softmax')  # Deux classes : fake et real
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def evaluate_model(model, test_data, history):
    # Evaluate the performance in the test set
    test_loss, test_acc = model.evaluate(test_data)
    print(f"Test Accuracy: {test_acc}, Test Loss: {test_loss}")

    # Generate plot and graph
    plot_training_history(history)


def plot_training_history(history):
    # Graphique de précision et perte
    acc = history.history['accuracy']
    loss = history.history['loss']
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 4))

    # Précision
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'b-', label='Training Accuracy')
    plt.title('Training Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Perte
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'r-', label='Training Loss')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.savefig("metrics/graphs/accuracy_loss_plot.png")
    plt.show()