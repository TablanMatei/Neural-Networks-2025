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



#Convertire lists in np arrays
train_data = np.array(train_data)
train_labels = np.array(train_labels)
test_data = np.array(test_data)

''' PENTRU ASSIGNMENTUL 3 DE AICI ....'''
def relu(Z):
    return np.maximum(0, Z)

def relu_derivative(Z):
    return (Z > 0).astype(float)

#MLP: Imput-Hidden_Layer-Output: 784-100-10 neuroni
hidden_size = 100

# Layer 1:
W1 = np.random.randn(784, hidden_size) * np.sqrt(2.0 / 784)  # He initialization pentru ReLU
b1 = np.zeros(hidden_size)

# Layer 2:
W2 = np.random.randn(hidden_size, 10) * np.sqrt(2.0 / hidden_size)
b2 = np.zeros(10)

'''..... PANA AICI '''

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
epochs = 150  # = o trecere compelta prin toate datele de training
batch_size = 100 # =un subset mic de date procesat o data
lambda_reg = 0.0000000000000000001  # L2 regularization strength



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

        ''' PENTRU ASSIGNMENTUL 3 DE AICI ....'''
        # Forward propagation Layer 1:
        Z1 = np.dot(X_batch, W1) + b1
        A1 = relu(Z1)

        # Forward propagation Layer 2:
        Z2 = np.dot(A1, W2) + b2
        Y_pred = softmax(Z2)
        '''..... PANA AICI '''

        ''' PENTRU ASSIGNMENTUL 3 DE AICI .... L2 REGULARIZATION'''
        # Calcul loss cu L2 regularization
        batch_loss = cross_entropy_loss(Y_pred, Y_batch)

        # Adauga L2 penalty: (λ/2) * (||W1||² + ||W2||²)
        l2_penalty = (lambda_reg / 2) * (np.sum(W1 ** 2) + np.sum(W2 ** 2))
        batch_loss_with_reg = batch_loss + l2_penalty

        epoch_loss += batch_loss_with_reg
        '''..... PANA AICI '''

        ''' PENTRU ASSIGNMENTUL 3 DE AICI ....'''
        # Backward propagation Layer 2
        # Gradient pentru output layer:
        dZ2 = Y_batch - Y_pred
        dW2 = np.dot(A1.T, dZ2) / X_batch.shape[0]  - lambda_reg * W2
        db2 = np.mean(dZ2, axis=0)

        # Backward propagation Layer 1
        # Chain rule:
        dA1 = np.dot(dZ2, W2.T)
        dZ1 = dA1 * relu_derivative(Z1)
        dW1 = np.dot(X_batch.T, dZ1) / X_batch.shape[0]  - lambda_reg * W1
        db1 = np.mean(dZ1, axis=0)

        #Update weights si biases pentru ambele layers
        W2 = W2 + learning_rate * dW2
        b2 = b2 + learning_rate * db2
        W1 = W1 + learning_rate * dW1
        b1 = b1 + learning_rate * db1
        '''..... PANA AICI '''

        num_batches += 1

    # Afisare loss la fiecare epoch
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {epoch_loss / num_batches:.4f}")

''' PENTRU ASSIGNMENTUL 3 DE AICI ....'''
# Predictii test:
Z1_test = np.dot(test_data, W1) + b1
A1_test = relu(Z1_test)

Z2_test = np.dot(A1_test, W2) + b2
Y_test_pred = softmax(Z2_test)

predictions = np.argmax(Y_test_pred, axis=1)

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

