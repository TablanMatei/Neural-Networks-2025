from torch.utils.data import Dataset


import pickle
import os
import pandas as pd
import numpy as np

class ExtendedMNISTDataset(Dataset):
     def __init__(self, root: str = "data", train: bool = True):
        file = "extended_mnist_test.pkl"
        if train:
            file = "extended_mnist_train.pkl"
        file = os.path.join(root, file)
        with open(file, "rb") as fp:
            self.data = pickle.load(fp)

     def __len__(self, ) -> int:
        return len(self.data)

     def __getitem__(self, i : int):
        return self.data[i]


train_data = []
train_labels = []
for image, label in ExtendedMNISTDataset(train=True):
    train_data.append(image.flatten())
    train_labels.append(label)



'''''DE AICI PANA.............'''

test_data = []
for image, label in ExtendedMNISTDataset(train=False):
    test_data.append(image.flatten())

train_data = np.array(train_data)
train_labels = np.array(train_labels)
test_data = np.array(test_data)

train_data = train_data.astype('float32') / 255.0
test_data = test_data.astype('float32') / 255.0

'''..................AICI'''



from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(
    hidden_layer_sizes=(256,128,64),
    alpha=1e-5,
    activation='relu',
    solver="sgd",
    verbose=10,
    random_state=1,
    learning_rate_init=0.1,
    learning_rate='adaptive',
    early_stopping=True,
    validation_fraction=0.1,
)
print("\nStarting training...")
mlp.fit(train_data, train_labels)
train_accuracy = mlp.score(train_data, train_labels)
print(f"\nTrain Accuracy: {train_accuracy:.4f}")

predictions = mlp.predict(test_data)

# This is how you prepare a submission for the competition
predictions_csv = {
    "ID": [],
    "target": [],
}

for i, label in enumerate(predictions):
    predictions_csv["ID"].append(i)
    predictions_csv["target"].append(label)

df = pd.DataFrame(predictions_csv)
df.to_csv("submission.csv", index=False)