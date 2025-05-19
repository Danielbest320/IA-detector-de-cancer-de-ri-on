# train.py
import torch
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
import torch.nn as nn

from dataset import FFTImageDataset
from Model import get_modified_resnet18

def main():
    root_dir = r"C:\Users\ivanv\Desktop\ProyectoMatematicasEspeciales\ImagenesCancer"

    dataset = FFTImageDataset(root_dir=root_dir, img_size=224)

    print("Clases encontradas:", dataset.classes)
    print("Número total de imágenes:", len(dataset))
    print("Primeros 5 ejemplos (ruta, etiqueta):")
    for i in range(min(5, len(dataset))):
        print(dataset.image_paths[i])

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_modified_resnet18(num_classes=2).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(10):
        model.train()
        running_loss = 0.0
        correct = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()

        train_acc = correct / len(train_loader.dataset)
        print(f"Época {epoch+1}, Pérdida entrenamiento: {running_loss:.4f}, Precisión entrenamiento: {train_acc:.4f}")

        model.eval()
        val_loss = 0.0
        val_correct = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                val_correct += (outputs.argmax(1) == labels).sum().item()
        val_acc = val_correct / len(val_loader.dataset)
        print(f"Validación - Pérdida: {val_loss:.4f}, Precisión: {val_acc:.4f}")
        
        
        
    # Después del entrenamiento
    torch.save(model.state_dict(), "modelo_resnet_fft.pth")
    print("Modelo guardado en 'modelo_resnet_fft.pth'")

if __name__ == "__main__":
    main()
    
    


