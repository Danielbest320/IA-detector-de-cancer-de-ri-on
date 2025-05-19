import torch
import cv2
import numpy as np
from Model import get_modified_resnet18  # Asegúrate que está en el mismo directorio o en PYTHONPATH

def prepare_image(path, img_size=224):
    
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        print(f"Error: no se puede leer la imagen en {path}")
        return None  # Sale si no se puede leer la imagen
    # Lee imagen en escala de grises
    img = cv2.resize(img, (img_size, img_size))
    img = img.astype(np.float32) / 255.0

    # Calcula magnitud FFT normalizada
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)
    magnitude = np.log(1 + magnitude)
    magnitude = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min())

    # Apila imagen original y FFT como canales
    stacked = np.stack([img, magnitude], axis=0)  # (2, H, W)
    tensor = torch.tensor(stacked, dtype=torch.float32).unsqueeze(0)  # (1, 2, H, W)
    return tensor

def predict(image_path, model_path="modelo_resnet_fft.pth", device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Carga modelo y pesos
    model = get_modified_resnet18(num_classes=2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Prepara imagen
    img_tensor = prepare_image(image_path).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.softmax(output, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()

    classes = ['Normal', 'Tumor']
    print(f"Predicción: {classes[pred_class]}")
    print(f"Probabilidades: {probs.cpu().numpy()}")

if __name__ == "__main__":
    ruta_imagen = r"C:\Users\ivanv\Desktop\ProyectoMatematicasEspeciales\imagen4.jpg"

    predict(ruta_imagen)
