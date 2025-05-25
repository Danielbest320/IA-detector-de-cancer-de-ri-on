#Importar librerias necesarias para trabajar con PyTorch
from torchvision.models import resnet18 #Modelo ResNet-18 pre-entrenado
import torch.nn as nn #Modulo de PyTorch para la construir redes neuronales

def get_modified_resnet18(num_classes=2):
    #Se carga rl modelo ResNet-18 pre-entrenado y se modifica
    #Debido a que originalmete espera 3 canales de entrada RGB
    model = resnet18(pretrained=True)

    #Se cambia para que acepte 2 canales:
    #1. La imagen en escala de grises
    #2. Su Transformada Rapida de Fourier (FFT)
    #Los parametros out_channels=64, kernel_size=7, stride=2, padding=3, bias=False
    #se mantienen iguales a los de la capa conv1 original de ResNet-18 para consistencia en el modelo.
    model.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)

    #Se reemplaza la capa fc para para que tengan el mismo numero de caracteristicas que la de entrada
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
