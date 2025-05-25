from torch.utils.data import Dataset #Clase base usada para datasets personalizados en PyTorch
import os #Para interartuar con los archivos
import cv2 #Para el procesamiento de imagen
import numpy as np #Para las operaciones numericas
import torch #Biblioteca principal para PyTorc

class FFTImageDataset(Dataset):
    """
    Dataset usado para cargar las imagenes, calcular la FFT
    y apilar la imagen original en escala de grises con su
    FFT correspondiente como una entrada de 2 canales
    """

    
    def __init__(self, root_dir, transform=None, img_size=224):
        self.root_dir = root_dir #Directorio que contiene las subcarpetas de las imagenes
        self.transform = transform #Se almacenara la transformada de muestra
        self.img_size = img_size #Dimensiones de la imagen
        #Obtener las clases y el nombre de sus subcarpetas ordenadas
        self.classes = sorted(os.listdir(root_dir))
        #Tupla para almacenar la ruta de la imagen y su etiqueta
        self.image_paths = []

        #Recorre cada clase con sus imagenes para construir la lista
        for label, class_name in enumerate(self.classes):
            class_dir = os.path.join(root_dir, class_name) #Ruta de la carpeta
            for filename in os.listdir(class_dir):
                # Considera los archivo que solo tengan estas extensiones de imagenes
                if filename.endswith(('.jpg', '.png')):
                    #Añade la ruta y la etiqueta
                    self.image_paths.append((os.path.join(class_dir, filename), label))

    #Calcula la magnitud logaritmica normalizada de la FFT en 2D
    #Espera resivir una imagen en escala de grises
    def compute_fft_magnitude(self, image):
        #1. Calcula la FFT de la imagen
        f = np.fft.fft2(image)
        #2. Centra el componente de la Frecuencia cero en el espetro
        fshift = np.fft.fftshift(f)
        #3. Calcula la magnitud del espetro de frecuencia
        # Esto debido a que los valores de FFT son complejos
        magnitude = np.abs(fshift)
        #4. Se aplica una transformada logaritmica para mejorar el contraste
        # Ademas de suma 1 para evitar algun logaritmo de cero
        magnitude = np.log(1 + magnitude)
        #5. Se normaliza la magnitud en un rango de 0 y 1
        magnitude = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min())
        #6.Se retorna la FFT procesada y normalizada
        return magnitude

    #Devuelve el numero total de muestras de imagenes del dataset
    def __len__(self):
        return len(self.image_paths)

    # Obtiene una nuestra del dataset dado un indice
    # Recibiendo como parametro el indice de la muestra a obtener
    def __getitem__(self, idx):
        #Se obtiene la ruta de la imagen y su etiqueta a partir del indice
        path, label = self.image_paths[idx]
        #Carga la imagen en escala de grises
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        #Redimenciona la imagen al tamaño especificado
        img = cv2.resize(img, (self.img_size, self.img_size))
        #Conviete la imagen a tipo float32 para poder normalizar sus valores
        #Estos quedaran en un rango de 0 a 1
        img = img.astype(np.float32) / 255.0
        #La imagen en escala de grises es el primer canal de la red neuronal

        #Calcula la magnitud de la FFT normalizada
        fft_mag = self.compute_fft_magnitude(img)
        #Esta es la segunda capa de la red

        #Convierte la imagen apilada y la etiqueta a tensores de PyTorch
        #Un tensor es la estructura de datos que se envia para el aprendizade automatico de la IA
        stacked = np.stack([img, fft_mag], axis=0)  # Shape: (2, H, W)

        #Retorna el primer tensor, como la imagen en 2 canales (la imagen en escala de grises y su FFT)
        #El segundo tensor, que es la etiqueta de la clase
        return torch.tensor(stacked, dtype=torch.float32), torch.tensor(label, dtype=torch.long)
