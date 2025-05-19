import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from Model import get_modified_resnet18

def prepare_image(path, img_size=224, show_images=False):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: no se puede leer la imagen en {path}")
        return None
    img = cv2.resize(img, (img_size, img_size))
    img_norm = img.astype(np.float32) / 255.0

    # FFT magnitud
    f = np.fft.fft2(img_norm)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)
    magnitude_log = np.log(1 + magnitude)
    magnitude_norm = (magnitude_log - magnitude_log.min()) / (magnitude_log.max() - magnitude_log.min())

    if show_images:
        fig, axs = plt.subplots(1, 2, figsize=(10,5))
        axs[0].imshow(img_norm, cmap='gray')
        axs[0].set_title('Imagen Original (Escala de Grises)')
        axs[0].axis('off')

        axs[1].imshow(magnitude_norm, cmap='gray')
        axs[1].set_title('Magnitud FFT (Log Normalizada)')
        axs[1].axis('off')

        plt.show()

    stacked = np.stack([img_norm, magnitude_norm], axis=0)
    tensor = torch.tensor(stacked, dtype=torch.float32).unsqueeze(0)
    return tensor

def predict(image_path, model_path="modelo_resnet_fft.pth", device=None, show_images=False):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_modified_resnet18(num_classes=2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    img_tensor = prepare_image(image_path, show_images=show_images)
    if img_tensor is None:
        return  # Error leyendo imagen

    img_tensor = img_tensor.to(device)

    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.softmax(output, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()

    classes = ['Normal', 'Tumor']
    print(f"Predicción: {classes[pred_class]}")
    print(f"Probabilidades: {probs.cpu().numpy()}")

if __name__ == "__main__":
    ruta_imagen = r"C:\Users\ivanv\Desktop\ProyectoMatematicasEspeciales\andreaaaa.jpg"
    predict(ruta_imagen, show_images=True)
