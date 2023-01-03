
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image

import torchvision.transforms as transforms


class YOLOv5Dataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None, mode='relative'):
        self.root_dir = root_dir
        self.transform = transform
        self.mode = mode  #'relative' ou 'absolute'
        self.images = []
        self.annotations = []

        #Récupère des fichiers d'images et d'annotations du répertoire
        for file in os.listdir(root_dir):
            if file.endswith(".jpg") or file.endswith(".png"):
                self.images.append(os.path.join(root_dir, file))
                annotation_file = file.replace(".jpg", ".txt").replace(".png", ".txt")
                self.annotations.append(os.path.join(root_dir, annotation_file))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        # Charge l'image
        image = cv2.imread(self.images[index])
        image_height, image_width, _ = image.shape

        # Charge les annotations et les traite au format attendu par le modèle
        with open(self.annotations[index], "r") as f:
            lines = f.readlines()
        annotations = []
        for line in lines:
            # Traite chaque ligne d'annotation
            x, y, w, h, class_id = line.strip().split()
            x = float(x)
            y = float(y)
            w = float(w)
            h = float(h)
            class_id = int(float(class_id))
            annotations.append((x, y, w, h, class_id))
            if self.mode == 'relative':
                # Convertit les coordonnées en mode relatif
                x /= image_width
                y /= image_height
                w /= image_width
                h /= image_height
            annotations.append((x, y, w, h, class_id))

        # Applique une éventuelle transformation sur l'image
        if self.transform:
            image = self.transform(image)

        print(annotations)

        return image, annotations

#Définis le modèle
class LicensePlateRecognitionModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(1024, 2048, kernel_size=3, stride=1, padding=1)
        self.conv9 = nn.Conv2d(2048, 4096, kernel_size=3, stride=1, padding=1)
        self.conv10 = nn.Conv2d(4096, num_classes, kernel_size=1, stride=1, padding=0)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.batchnorm1 = nn.BatchNorm2d(16)
        self.batchnorm2 = nn.BatchNorm2d(32)
        self.batchnorm3 = nn.BatchNorm2d(64)
        self.batchnorm4 = nn.BatchNorm2d(128)
        self.batchnorm5 = nn.BatchNorm2d(256)
        self.batchnorm6 = nn.BatchNorm2d(512)
        self.batchnorm7 = nn.BatchNorm2d(1024)
        self.batchnorm8 = nn.BatchNorm2d(2048)
        self.batchnorm9 = nn.BatchNorm2d(4096)

    def forward(self, x):
        x = self.maxpool(self.batchnorm1(F.relu(self.conv1(x))))
        x = self.maxpool(self.batchnorm2(F.relu(self.conv2(x))))
        x = self.maxpool(self.batchnorm3(F.relu(self.conv3(x))))
        x = self.maxpool(self.batchnorm4(F.relu(self.conv4(x))))
        x = self.maxpool(self.batchnorm5(F.relu(self.conv5(x))))
        x = self.maxpool(self.batchnorm6(F.relu(self.conv6(x))))
        x = self.maxpool(self.batchnorm7(F.relu(self.conv7(x))))
        x = self.maxpool(self.batchnorm8(F.relu(self.conv8(x))))
        x = self.maxpool(self.batchnorm9(F.relu(self.conv9(x))))
        x = self.conv10(x)
        return x
def init_weights(m):
    if type(m) == nn.Conv2d:
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif type(m) == nn.BatchNorm2d:
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

num_classes = 36
num_epochs = 2
dataset = YOLOv5Dataset(root_dir=os.getcwd()+"/../data", transform=transforms.ToTensor(), mode='relative')



#Initialisation modèle
model = LicensePlateRecognitionModel(num_classes)
model.apply(init_weights)
optimizer = optim.SGD(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

#Séparation des données d'entraînement et de validation à partir du data loader
dataset_size = len(dataset)
train_size = int(0.8 * dataset_size)
val_size = dataset_size - train_size
train_dataset, validation_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

#Création dataloader
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=32, shuffle=False)

model.train()
#Entrainement du modèle
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_dataloader):
        optimizer.zero_grad()

        # Créez un tenseur d'entiers avec l'annotation
        """labels = [label[0] for label in labels]
        labels = [label.view(-1) for label in labels]
        labels = torch.cat(labels, dim=0)
        # Sélectionnez le label associé à chaque image
        labels = labels[range(images.size(0))]"""
        # Prediction
        outputs = model(images)
        labels_tensor = torch.Tensor(np.argmax(labels))
        # Calcul de perte
        loss = criterion(outputs, labels_tensor)

        # Back Propagation
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_dataloader)}], Loss: {loss.item():.4f}')

        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in validation_dataloader:
                labels = [label[0] for label in labels]
                labels = [label.view(-1) for label in labels]
                labels = torch.cat(labels, dim=0)
                # Sélection label associé à chaque image
                labels = labels[range(images.size(0))]
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            print(f'Accuracy of the model on the validation images: {100 * correct / total} %')