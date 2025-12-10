import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import os


# --- 1. PRZYGOTOWANIE DANYCH (potrzebne do wyświetlenia przykładów testowych) ---

def gaussian_noise(image, var=0.1):
    sigma = var ** 0.5
    gaussian = torch.randn_like(image) * sigma
    noisy_image = image + gaussian
    noisy_image = torch.clamp(noisy_image, 0.0, 1.0)
    return noisy_image


print("Ładowanie danych testowych...")
transform = transforms.ToTensor()
# Pobieramy tylko testowe, żeby było szybciej (treningowe nie są potrzebne do wyświetlania wyników)
test_dataset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

test_clean_all = torch.stack([img for img, _ in test_dataset])
test_noisy_all = torch.stack([gaussian_noise(img) for img in test_clean_all])


# --- 2. DEFINICJA MODELU (Musi być taka sama jak przy treningu) ---

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        # Enkoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Dekoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# --- 3. INICJALIZACJA I WCZYTANIE WAG ---

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Używane urządzenie: {device}")

model = AutoEncoder().to(device)

# Sprawdzenie czy plik istnieje
weights_path = 'autoencoder.pth'
if os.path.exists(weights_path):
    # map_location=device pozwala wczytać model trenowany na CPU na GPU i odwrotnie
    model.load_state_dict(torch.load(weights_path, map_location=device))
    print("Sukces: Wczytano wytrenowany model.")
else:
    print(f"BŁĄD: Nie znaleziono pliku {weights_path}. Upewnij się, że uruchamiasz skrypt w dobrym folderze.")
    exit()

model.eval()  # Przełączenie w tryb ewaluacji (wyłącza dropouty, batchnormy itp.)

# --- 4. WIZUALIZACJA WYNIKÓW ---

offset = 1002  # Przesunięcie, żeby zobaczyć inne obrazki niż na początku

# A. Oryginalne (Czyste)
print("Generowanie wykresów...")
plt.figure(figsize=(12, 4))
plt.suptitle("Real Test Images (Clean)")
for i in range(9):
    plt.subplot(1, 9, i + 1)
    plt.imshow(test_clean_all[i + offset].squeeze().numpy(), cmap='gray')
    plt.axis('off')
plt.show()

# B. Zaszumione (Wejście)
plt.figure(figsize=(12, 4))
plt.suptitle("Noisy Test Images (Input)")
for i in range(9):
    plt.subplot(1, 9, i + 1)
    plt.imshow(test_noisy_all[i + offset].squeeze().numpy(), cmap='gray')
    plt.axis('off')
plt.show()

# C. Odszumione przez model (Wyjście)
plt.figure(figsize=(12, 4))
plt.suptitle("Reconstructed Images (Output from Autoencoder)")
with torch.no_grad():  # Wyłączamy obliczanie gradientów dla szybkości
    for i in range(9):
        plt.subplot(1, 9, i + 1)

        # Przygotowanie pojedynczego obrazka do modelu (dodanie wymiaru batcha)
        noisy_image = test_noisy_all[i + offset].unsqueeze(0).to(device)

        # Puszczenie przez sieć
        output = model(noisy_image)

        # Konwersja z powrotem na format do wyświetlenia
        op_image = output[0].cpu().squeeze().numpy()

        plt.imshow(op_image, cmap='gray')
        plt.axis('off')
plt.show()