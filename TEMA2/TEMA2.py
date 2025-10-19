import pickle
import os
import pandas as pd
import numpy as np

train_file = "data/extended_mnist_train.pkl"
test_file = "data/extended_mnist_test.pkl"

with open(train_file, "rb") as fp:
    train = pickle.load(fp)

with open(test_file, "rb") as fp:
    test = pickle.load(fp)

train_data = []
train_labels = []
for image, label in train:
    train_data.append(image.flatten()/ 255.0)  # ptr normalizare
    train_labels.append(label)



test_data = []
for image, label in test:
    test_data.append(image.flatten()/ 255.0)  # ptr normalizare


''' EU AM DE IMPLEMENTAT DE AICI ....'''
#Convertire lists in np arrays
train_data = np.array(train_data)
train_labels = np.array(train_labels)
test_data = np.array(test_data)


# Initializare W (784, 10) si b (10,) - Xavier initialization
np.random.seed(50)
W = np.random.randn(784, 10) * np.sqrt(1.0 / 784)  # scalare cu sqrt(1/n_inputs)
b = np.zeros(10)


# Convertire train_labels la one-hot encoding (m, 10)
# Ex: Label 7 devine [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
m = len(train_labels)
train_labels_onehot = np.zeros((m, 10))
train_labels_onehot[np.arange(m), train_labels] = 1


# Definire functie softmax
def softmax(Z):
    exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))  # normalizare
    return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)   # e^(z_i) / Σ e^(z_j)

# Definire functie cross-entropy loss
# Loss = -Σ y_i log(ŷ_i)
# Exemplu:
# y_true = [0, 0, 1, 0, ...]  # clasa 2
# y_pred = [0.1, 0.2, 0.6, 0.1, ...]  # softmax predictions

def cross_entropy_loss(y_pred, y_true):
    m = y_pred.shape[0]  #shape[0]=nr de randuri, shape[1]=nr de coloane
    epsilon = 1e-10
    loss = -np.sum(y_true * np.log(y_pred + epsilon)) / m
    return loss


# Hyperparametri
learning_rate = 0.2  # μ din formule, luat din MLP Classifierul dat
epochs = 120  # = o trecere compelta prin toate datele de training
batch_size = 128 # =un subset mic de date procesat o data

# Loop antrenare
for epoch in range(epochs):
    indices = np.arange(m)
    np.random.shuffle(indices) #Amesteca random cifrele
    train_data_shuffled = train_data[indices]
    train_labels_shuffled = train_labels_onehot[indices]

    epoch_loss = 0
    num_batches = 0

    # Loop prin batch-uri
    for i in range(0, m, batch_size):
        # Extrage batch-ul curent
        X_batch = train_data_shuffled[i: i + batch_size]
        Y_batch = train_labels_shuffled[i: i + batch_size]

        # Forward propagation:
        Z = np.dot(X_batch, W) + b    # shape: (batch_size, 10)

        # Aplicare softmax:
        Y_pred = softmax(Z)

        # Calcul loss
        batch_loss = cross_entropy_loss(Y_pred, Y_batch)
        epoch_loss += batch_loss


        # Backward propagation: gradient dW și db
        # Gradient = diferenta dintre predictii si target
        gradient = Y_batch - Y_pred

        #X_batch.T = transpusa
        dW = np.dot(X_batch.T, gradient) / X_batch.shape[0]  # shape: (784, 10)
        db = np.mean(gradient, axis=0)  # shape: (10,)

        ### Update weights si biases
        W = W + learning_rate * dW
        b = b + learning_rate * db

        num_batches += 1

    # Afișare loss la fiecare epoch
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {epoch_loss / num_batches:.4f}")

# Predictii test: forward propagation + argmax pe axis=1
Z_test = np.dot(test_data, W) + b
Y_test_pred = softmax(Z_test)
predictions = np.argmax(Y_test_pred, axis=1)  # shape: (num_test,)

my_predictions = predictions.copy()
'''..... PANA AICI '''

from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(hidden_layer_sizes=(40,),
    alpha=1e-4,
    solver="sgd",
    verbose=10,
    random_state=1,
    learning_rate_init=0.2
)
mlp.fit(train_data, train_labels)

mlp.score(train_data, train_labels)


predictions = mlp.predict(test_data)

predictions_csv = {
    "ID": [],
    "target": [],
}

for i, label in enumerate(my_predictions): #Predictiile mele
    predictions_csv["ID"].append(i)
    predictions_csv["target"].append(label)

df = pd.DataFrame(predictions_csv)
df.to_csv("submission.csv", index=False)

#KAGGLE SCORE: 0.92680