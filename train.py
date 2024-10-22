# IMPORTS #

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os, random
import pandas as pd

## Imports for plotting
from matplotlib.colors import to_rgba
import seaborn as sns
sns.set_theme('notebook', style='whitegrid')

## Progress bar
from tqdm.notebook import tqdm


import torch
torch.manual_seed(42) # Setting the seed

print("\007")

class WikiArtImageDataset(Dataset):
    def __init__(self, txt, csv_file, img_dir, transform=None):
        self.classes = pd.read_csv(txt, sep=" ", header=None)
        self.classes.columns = ["id", "class_name"]
        self.annotations = pd.read_csv(csv_file, header=None)
        self.annotations.columns = ["img_path", "class_id"]
        self.annotations = self.annotations.sample(frac=0.1)
        self.img_dir = img_dir
        self.transform = transform
        # ENCODING STRINGS IN DICT TO DECODE THEM WHEN NEEDED
        self.enc = {name: id for id, name in self.classes.itertuples(index=False)}
        # Create a reverse mapping for integer labels back to string labels
        self.dec = {idx: label for label, idx in self.enc.items()}
                                                                         
    def __len__(self):
        return len(self.annotations)

    # Applied each time batches are returned by DataLoader, executing all the "preprocessing"
    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        image = Image.open(img_path).convert("RGB")
        label = self.annotations.iloc[index, 1]
        # Convert string label to integer
        if self.transform:
            image = self.transform(image)

        return image, label

# Paths to the data
classes = './data/wikiart/wikiart_csv/style_class.txt'
train_csv = './data/wikiart/wikiart_csv/style_train.csv'  # Path to your CSV file
val_csv = './data/wikiart/wikiart_csv/style_val.csv'  # Path to your CSV file                                                                                                                                                                                                                                                                                                                                                       
img_dir = './data/wikiart/wikiart_img/'       # Directory with all the images

# Define image transformations
# (ex.: trying to use padding instead of resizing)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize the images to 224x224 without preserving aspect ratio, i.e., squishing the image
    transforms.ToTensor(),
]) 
# Common practice to normalize data, especially when using pre-trained models which were themselves trained on normalized data (as RESNet)
# Create the datasets
train_dataset, val_dataset = WikiArtImageDataset(txt=classes, csv_file=train_csv, img_dir=img_dir, transform=transform), WikiArtImageDataset(txt=classes, csv_file=val_csv, img_dir=img_dir, transform=transform)

# Create DataLoader for train and validation datasets
batch_size = 128
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

# Example: Iterate through the training data
images, labels = next(iter(train_loader))
print(f"Train - images shape: {images.shape}, labels shape: {labels.shape}")

# Example: Iterate through the validation data
images, labels = next(iter(val_loader))
print(f"Validation - images shape: {images.shape}, labels shape: {labels.shape}")


class CustomCNN(nn.Module):
    def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(44944, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 27)

    def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = torch.flatten(x, 1) # flatten all dimensions except batch
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x
        

model_0 = CustomCNN()

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
torch.backends.cudnn.benchmark = True
model_0.to(device)

def train (train_data: torch.Tensor, val_data: torch.Tensor, loss_fn, opt, model, epochs=10):
    scaler = torch.amp.GradScaler("cuda")  # Initialize gradient scaler
    for epoch in range(epochs): 
        model.train()
        length = len(train_data)
        epoch_loss = 0 
        correct = 0
        num_batch = 0
        for images, labels in train_data:
            images, labels = images.to(device), labels.to(device)
            num_batch += 1
            opt.zero_grad()
            with torch.amp.autocast("cuda"):
                logits = model(images)
                loss = loss_fn(logits, labels)

            scaler.scale(loss).backward()
            scaler.stepr(opt)
            scaler.update()
            epoch_loss += loss
            
            probabilities = torch.softmax(logits, dim=1)
            label_pred = torch.argmax(probabilities, dim=1)
            correct += (label_pred == labels).sum().item()
            if num_batch % 100 == 0:
                print(num_batch)
        train_loss_rate = epoch_loss / length
        train_accuracy = correct * 100 / length
        model.eval()
        val_epoch_loss = 0
        val_correct = 0
        with torch.inference_mode():
            for val_image, val_label in val_data:
                val_image, val_label = val_image.to(device), val_label.to(device)

                val_output = model(val_image)
                val_loss = loss_fn(val_output, val_label)
                val_epoch_loss += val_loss

                val_pred = torch.softmax(val_output, dim=1).argmax(dim=1)
                val_correct += (val_pred == val_label).sum().item()
            
        val_loss_rate = val_epoch_loss / len(val_data)
        val_accuracy = val_correct * 100 / len(val_data)

        print(f"Epoch [{epoch+1}/{epochs}], "
              f"Train Loss: {train_loss_rate:.4f}, Train Accuracy: {train_accuracy:.2f}%, "
              f"Val Loss: {val_loss_rate:.4f}, Val Accuracy: {val_accuracy:.2f}%")
        

loss = torch.nn.CrossEntropyLoss()
opt = optim.Adam(params=model_0.parameters(),
                 lr=0.01)
train(train_loader, val_loader, loss, opt, model_0)