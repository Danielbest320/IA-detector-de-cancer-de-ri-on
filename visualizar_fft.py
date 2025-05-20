import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

# Importar el modelo (asegúrate de que estos archivos estén en el mismo directorio)
from Model import get_modified_resnet18

def compute_fft_magnitude(image):
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)
    magnitude = np.log(1 + magnitude)
    magnitude = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min())
    return magnitude

def compute_fft_1d_spectrum(image):
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)
    avg_amplitude = np.mean(magnitude, axis=0)

    freqs = np.fft.fftfreq(image.shape[1])
    freqs = np.fft.fftshift(freqs)
    freqs = np.abs(freqs * 10000)  # Escala simulada a Hz

    return freqs, avg_amplitude

def load_model_and_predict(image_path, model_path="modelo_resnet_fft.pth"):
    """
    Carga el modelo entrenado y realiza predicción sobre una imagen
    Procesa la imagen de la misma manera que el dataset del entrenamiento
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Cargar el modelo
    model = get_modified_resnet18(num_classes=2).to(device)
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print(f"Modelo cargado desde: {model_path}")
    else:
        print(f"No se encontró el modelo en: {model_path}")
        return None, None
    
    # Procesar la imagen de la misma manera que FFTImageDataset
    # Cargar imagen en escala de grises
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (224, 224))
    img = img.astype(np.float32) / 255.0
    
    # Calcular FFT
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)
    magnitude = np.log(1 + magnitude)
    
    # Normalizar
    magnitude = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min())
    
    # Combinar imagen original y FFT en 2 canales
    combined = np.stack([img, magnitude], axis=0)  # Shape: (2, 224, 224)
    
    # Convertir a tensor y añadir dimensión de batch
    input_tensor = torch.from_numpy(combined).unsqueeze(0).to(device)  # Shape: (1, 2, 224, 224)
    
    # Realizar predicción
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = F.softmax(outputs, dim=1)
        predicted_class = torch.argmax(outputs, dim=1).item()
    
    return predicted_class, probabilities[0]

def visualize_fft(image_path, model_path="modelo_resnet_fft.pth"):
    if not os.path.exists(image_path):
        print("Ruta no válida:", image_path)
        return

    # Cargar imagen para visualización FFT
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (224, 224))
    img = img.astype(np.float32) / 255.0

    # FFT 2D
    fft_mag = compute_fft_magnitude(img)

    # FFT 1D
    freqs, avg_amplitude = compute_fft_1d_spectrum(img)

    # Realizar predicción con el modelo
    pred_class, probs = load_model_and_predict(image_path, model_path)
    
    # Mostrar imagen y espectro 2D
    plt.figure(figsize=(15, 8))
    
    # Imagen original
    plt.subplot(2, 3, 1)
    img_rgb = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    plt.title("Imagen Original")
    plt.axis('off')

    # Imagen en escala de grises
    plt.subplot(2, 3, 2)
    plt.imshow(img, cmap='gray')
    plt.title("Imagen en Escala de Grises")
    plt.axis('off')

    # Magnitud FFT 2D
    plt.subplot(2, 3, 3)
    plt.imshow(fft_mag, cmap='inferno')
    plt.title("Magnitud FFT (2D)")
    plt.axis('off')

    # Perfil de intensidad (como onda)
    plt.subplot(2, 3, 4)
    row = img[112]
    plt.plot(row)
    plt.title("Perfil de Intensidad (Fila Central)")
    plt.xlabel("Pixel")
    plt.ylabel("Intensidad")
    plt.grid(True)

    # Espectro 1D (frecuencia vs amplitud)
    plt.subplot(2, 3, 5)
    plt.plot(freqs, avg_amplitude)
    plt.xscale('log')
    plt.ylim(0, np.max(avg_amplitude))
    plt.xlim(10, np.max(freqs))
    plt.xlabel("Frecuencia (Hz)")
    plt.ylabel("Amplitud")
    plt.title("Espectro de Frecuencia 1D")
    plt.grid(True, which='both', axis='both', linestyle='--', linewidth=0.5)

    # Resultados de predicción
    plt.subplot(2, 3, 6)
    classes = ['Normal', 'Tumor']
    if pred_class is not None and probs is not None:
        # Crear gráfico de barras para las probabilidades
        colors = ['green', 'red']
        bars = plt.bar(classes, probs.cpu().numpy(), color=colors, alpha=0.7)
        plt.title(f"Predicción del Modelo\nClase predicha: {classes[pred_class]}")
        plt.ylabel("Probabilidad")
        plt.ylim(0, 1)
        
        # Añadir valores sobre las barras
        for bar, prob in zip(bars, probs.cpu().numpy()):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{prob:.3f}', ha='center', va='bottom')
        
        # Resaltar la clase predicha
        bars[pred_class].set_edgecolor('black')
        bars[pred_class].set_linewidth(2)
    else:
        plt.text(0.5, 0.5, 'Modelo no disponible', 
                ha='center', va='center', transform=plt.gca().transAxes)
        plt.title("Sin Predicción")
    
    plt.tight_layout()
    plt.show()

    # Imprimir resultados
    if pred_class is not None and probs is not None:
        print(f"\n{'='*50}")
        print(f"RESULTADOS DE LA PREDICCIÓN")
        print(f"{'='*50}")
        print(f"Clase predicha: {classes[pred_class]}")
        print(f"Probabilidades:")
        for i, (clase, prob) in enumerate(zip(classes, probs.cpu().numpy())):
            print(f"  {clase}: {prob:.4f} ({prob*100:.2f}%)")
        print(f"Confianza: {torch.max(probs).item():.4f}")
        print(f"{'='*50}")
    else:
        print("No se pudo realizar la predicción. Verifica que el modelo esté entrenado.")

# Función adicional para comparar múltiples imágenes
def compare_multiple_images(image_paths, model_path="modelo_resnet_fft.pth"):
    """
    Compara las predicciones de múltiples imágenes
    """
    results = []
    classes = ['Normal', 'Tumor']
    
    for img_path in image_paths:
        if os.path.exists(img_path):
            pred_class, probs = load_model_and_predict(img_path, model_path)
            if pred_class is not None:
                results.append({
                    'path': img_path,
                    'predicted_class': classes[pred_class],
                    'probabilities': probs.cpu().numpy(),
                    'confidence': torch.max(probs).item()
                })
    
    # Mostrar comparación
    if results:
        print(f"\n{'='*80}")
        print(f"COMPARACIÓN DE MÚLTIPLES IMÁGENES")
        print(f"{'='*80}")
        for i, result in enumerate(results):
            print(f"{i+1}. {os.path.basename(result['path'])}")
            print(f"   Predicción: {result['predicted_class']}")
            print(f"   Confianza: {result['confidence']:.4f}")
            print(f"   Normal: {result['probabilities'][0]:.4f}, Tumor: {result['probabilities'][1]:.4f}")
            print("-" * 80)

# Ejecutar
if __name__ == "__main__":
    # Ejemplo con una sola imagen
    #ruta = r"C:\Users\ASUS\Desktop\Especiales\ProyectoEspeciales\IA-detector-de-cancer-de-ri-on\imagen4.jpg"
    #visualize_fft(ruta)
    #
    # Ejemplo con múltiples imágenes (opcional)
    rutas_multiples = [
        r"C:\Users\ASUS\Desktop\Especiales\ProyectoEspeciales\IA-detector-de-cancer-de-ri-on\imagen9.jpg",
        r"C:\Users\ASUS\Desktop\Especiales\ProyectoEspeciales\IA-detector-de-cancer-de-ri-on\imagen3.jpg",
        r"C:\Users\ASUS\Desktop\Especiales\ProyectoEspeciales\IA-detector-de-cancer-de-ri-on\imagen4.jpg"
    ]
    visualize_fft(rutas_multiples[2])
    compare_multiple_images(rutas_multiples)