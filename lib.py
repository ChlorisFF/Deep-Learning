import pickle
import matplotlib
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
import torch.nn as nn
from torch.utils.data import DataLoader as pt_DataLoader, TensorDataset
from collections import defaultdict
import torch
import clip
from torchvision import models, transforms
import numpy as np
from PIL import Image
from tqdm import tqdm
import random
matplotlib.use('TkAgg')

class DataLoader:

    def __init__(self, dataset_path="cifar-100-python"):
        self.dataset_path = dataset_path
        self.train_data = None
        self.test_data = None
        self.meta_data = None
        self.load_data()

    def unpickle(self, file):
        with open(file, 'rb') as fo:
            data = pickle.load(fo, encoding='bytes')
        return data

    def load_data(self):
        self.train_data = self.unpickle(self.dataset_path + "/train")
        self.test_data = self.unpickle(self.dataset_path + "/test")
        self.meta_data = self.unpickle(self.dataset_path + "/meta")

    def get_data(self):
        return self.train_data, self.test_data, self.meta_data

    def get_names(self):
        return [name.decode('utf-8') for name in self.meta_data[b'fine_label_names']]


# Τesting ------------------------------------------------------------------------------------------------------

def displayDataTest(X_train, y_train, X_test, y_test):

    print(f"Training data shape: {X_train.shape}")
    print(f"Training labels shape: {len(y_train)}")
    print(f"Test data shape: {X_test.shape}")
    print(f"Test labels shape: {len(y_test)}")

    # Display the first image
    plt.imshow(X_train[0])
    plt.title(f"Label: {y_train[0]}")
    plt.show()


#  k-NN classifier----------------------------------------------------------------------------------------------

def filter_data_by_classes(X_train, y_train, X_test, y_test, selected_names, label_names):

    selected_indices = [label_names.index(cls) for cls in selected_names]

    train_mask = np.isin(y_train, selected_indices)
    test_mask = np.isin(y_test, selected_indices)

    X_train_filtered, y_train_filtered = X_train[train_mask], y_train[train_mask]
    X_test_filtered, y_test_filtered = X_test[test_mask], y_test[test_mask]

    return X_train_filtered, y_train_filtered, X_test_filtered, y_test_filtered


def normalize_and_reshape_numpy(X_train, X_test):

    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train_flat)
    X_test_scaled = scaler.transform(X_test_flat)

    return X_train_scaled, X_test_scaled


def train_and_evaluate_knn(X_train, y_train, X_test, y_test, n_neighbors=3):
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print(f"Accuracy: {accuracy:.2f}")

    return accuracy


# Autoencoder ----------------------------------------------------------------------------------------------------------

def prepare_data_for_pytorch(X_train, X_test, batch_size=128):

    # To scale it we need to flatten it
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)

    scaler = MinMaxScaler() # here we are scaling
    X_train_scaled = scaler.fit_transform(X_train_flat)
    X_test_scaled = scaler.transform(X_test_flat)

    # Now we reshape as PyTorch wants it
    X_train_pT = X_train_scaled.reshape(-1, 3, 32, 32)
    X_test_pT = X_test_scaled.reshape(-1, 3, 32, 32)

    # We create the tensors
    X_train_tensor = torch.tensor(X_train_pT, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_pT, dtype=torch.float32)

    # We use the PyTorch dataloader
    train_loader = pt_DataLoader(TensorDataset(X_train_tensor), batch_size=batch_size, shuffle=True)

    # Ensuring tensors and the model are on the same device. (I am using cpu)
    X_train_tensor = X_train_tensor.to(device)
    X_test_tensor = X_test_tensor.to(device)

    return train_loader, X_train_tensor, X_test_tensor


class Autoencoder(nn.Module):
    # This autoencoder extends PyTorch module and therefore have the same methods as a normal PyTorch module
    # Also I implement the forward function that defines how the data flows through the model
    def __init__(self, encoded_dim=64):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, encoded_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoded_dim, 64 * 4 * 4),
            nn.ReLU(),
            nn.Unflatten(1, (64, 4, 4)),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


def train_autoencoder(autoencoder, train_loader, optimizer, criterion, epochs=10):

    # Model should be on the available device
    autoencoder.to(device)

    for epoch in range(epochs):
        autoencoder.train()
        for batch in train_loader:
            inputs = batch[0].to(device)  # Move input data to the same device as the model
            optimizer.zero_grad()
            encoded, decoded = autoencoder(inputs)
            loss = criterion(decoded, inputs)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.detach().item():.4f}")


# Comparison ------------------------------------------------------------------------------------------

# used to check dataset balance
def group_data_by_class(X, y):
    data_by_class = defaultdict(list)

    for img, label in zip(X, y):
        data_by_class[label].append(img)

    for label in data_by_class:
        data_by_class[label] = np.array(data_by_class[label])

    return data_by_class


# -------------CLIP-----------------------------------------------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
model.eval().to(device)

def zero_shot_clip_predict(X, y_true, class_names):

    correct = 0
    total = len(X)
    X_shaped = X.reshape(-1, 32, 32, 3)  # Shape (samples, height, width, channels)

    prompts = [f"a photo of a {label}" for label in class_names]
    text_tokens = clip.tokenize(prompts).to(device)

    with torch.no_grad():
         for i in tqdm(range(total), desc="CLIP Zero-Shot Predict"):
            img = X_shaped[i]
            img = Image.fromarray((img * 255).astype(np.uint8)) if img.max() <= 1 else Image.fromarray(img.astype(np.uint8))
            image_input = preprocess(img).unsqueeze(0).to(device)

            logits_per_image, logits_per_text = model(image_input, text_tokens)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()

            #print(f"pred {class_names[pred]} = true {y_true[i]}")

            pred = probs.argmax()

            if class_names[pred] == y_true[i]:
                correct += 1

    acc = correct / total
    return acc

resnet50 = models.resnet50()
num_features = resnet50.fc.in_features
resnet50.fc = torch.nn.Linear(num_features, 10)  # 10 classes for the subset
resnet50.eval().to(device)

def resnet_predict(X, y_true, class_names):
    correct = 0
    total = len(X)

    # Preprocessing for ResNet-50 (resizing, cropping, normalization)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    X_shaped = X.reshape(-1, 32, 32, 3)

    with torch.no_grad():
        for i in tqdm(range(total), desc="ResNet-50 Prediction"):
            img = X_shaped[i]
            img = Image.fromarray(img.astype(np.uint8))

            image_input = preprocess(img).unsqueeze(0).to(device)

            outputs = resnet50(image_input)
            _, predicted = torch.max(outputs.data, 1)

            pred_class = class_names[predicted.item()]

            if pred_class == y_true[i]:
                correct += 1

    accuracy = correct / total
    return accuracy

# _________________________________________________________________

def pick_random_classes(class_list, num_classes=10):
    if len(class_list) < num_classes:
        raise ValueError("Not enough classes to choose from.")

    return random.sample(class_list, num_classes)


def print_model_performance(selected_names, accuracy_knn, autoencoder_knn, accuracy_original_knn, acc_clip, acc_res):
    print("Selected Classes".center(40, "-"))
    print(", ".join(selected_names))
    print("\n" + "Model Performance Comparison".center(40, "-"))
    print(f"{'Model':<20} | {'Accuracy':>10}")
    print("-" * 33)
    print(f"{'KNN':<20} | {accuracy_knn:>10.4f}")
    print(f"{'Autoencoder KNN':<20} | {autoencoder_knn:>10.4f}")
    print(f"{'KNN (Original)':<20} | {accuracy_original_knn:>10.4f}")
    print(f"{'CLIP':<20} | {acc_clip:>10.4f}")
    print(f"{'ResNet-50':<20} | {acc_res:>10.4f}")
    print("-" * 33)
