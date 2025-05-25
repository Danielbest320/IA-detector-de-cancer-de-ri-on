import torch #Biblioteca principal de PyTorch
from torch.utils.data import DataLoader, random_split #Utilizada para cargar los datos y division de datasets
import torch.optim as optim #Modulo de optimizadores para la red neuronal
import torch.nn as nn #Modulo para construir redes neuronales

#Importa las clases de los archivos previamente creados
from dataset import FFTImageDataset #El cargador de datos personalizado
from Model import get_modified_resnet18 #Obtiene el modelo ResNet-18 modificado

#Ejecuta el entrenamiento y validacion del modelo
def main():
    #--- 1. Configuracion del Dataset ---

    #Se define la direccion del documento en donde se encuentras las images y sus subcarpetas
    root_dir = r"C:\Users\ASUS\Desktop\Especiales\ProyectoEspeciales\IA-detector-de-cancer-de-ri-on\ImagenesCancer"

    #Se crea una instancia del dataset
    #Redimensiona el tamaño de las imagenes
    dataset = FFTImageDataset(root_dir=root_dir, img_size=224)

    #Se imprime la informacion del dataset para verificar que este correcto
    print("Clases encontradas:", dataset.classes)
    print("Número total de imágenes:", len(dataset))
    #Imprime las primeras 5 rutas de las images y sus etiquetas
    print("Primeros 5 ejemplos (ruta, etiqueta):")
    for i in range(min(5, len(dataset))):
        print(dataset.image_paths[i])

    #Se divide el dataset en conjuntos de datos de entrenamiento y validacion
    #Siendo el 80% para entrenamiento y el 20% para validacion
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    #Se crean DataLoaders para cada conjunto de entrenamiento y validacion
    #El DataLoader se encarga de cargar los datos en lote o batches
    #Se procesaran 32 imagenes a la vez y se mezclaran para evitar que estas entren en el mismo orden
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)


    #---2. Configuracion del modelo y el entorno de entrenamiento ---

    #Se selecciona el dispositivo de computo usado para el entrenamiento
    #Usara la GPU(cuda) si se encuentra disponible, si no usara la CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #Inicializacion del modelo ResNet-18 que modificamos
    #Se fija la salida para que el numero de clases sea 2
    #Osea que imagen con cancer y sin cancer
    model = get_modified_resnet18(num_classes=2).to(device)

    #Usamos el algoritmo de optimizacion Adam
    #model.parameters() le indica al optimizador que patrones debe ajustar
    #Se coloca la tasa de aprendizaje en 0.0001
    #Esta controla la magnitud de los ajustes que se realicen
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    #Se define la funcion de perdida, osea el que tanto se "equivoca" en las predicciones
    #CrossEntropyLoss se usa para problemas de clasificacion como el que tenemos
    criterion = nn.CrossEntropyLoss()


    #--- 3. Ciclo de entrenamiento y Validacion ---

    #Se define el numero de veces que se recorrera por completo el conjunto de los datos de entrenamiento
    #Para este caso sera un numero total de 10 veces
    for epoch in range(10):

        #Entrenamiento
        model.train() #Se pone el modelo en modo entrenamiento
        running_loss = 0.0  #Acomula la perdida de la validacion
        correct = 0 #Para contar las predicciones correctas

        #Se itera sobre los lotes de los datos de entrenamiento
        for inputs, labels in train_loader:
            #Mueve los datos y los etiqueda al dispositivo de entrada especificado
            inputs, labels = inputs.to(device), labels.to(device)

            # Se reinician los radiantes, para evitar que se acomulen de ciclos anteriores
            optimizer.zero_grad()
            # Realiza un paso hacia adelante (forward pass) a traves del modelo para obtener las predicciones
            outputs = model(inputs)
            #Calcula la perdida comparando con las predicciones (outputs), con las etiquetas reales (labels)
            loss = criterion(outputs, labels)
            #Realiza propagacion hacia atras (backward pass) para calcular los gradientes de la perdida
            loss.backward()
            #Optimiza los pesos del modelo utilizando el optimizador y los gradientes calculados
            optimizer.step() #Con esto se busca reducir la perdida

            #Suma la perdida del lote actual a la perdida acomulada de la epoca
            running_loss += loss.item()
            #Cuenta las predicciones correctas
            correct += (outputs.argmax(1) == labels).sum().item()

        #Se calcula la precision del entrenamiento
        train_acc = correct / len(train_loader.dataset)
        print(f"Época {epoch+1}, Pérdida entrenamiento: {running_loss:.4f}, Precisión entrenamiento: {train_acc:.4f}")

        #Validacion
        model.eval() #Se pone el modelo en modo de evaluacion
        val_loss = 0.0 #Acomula la perdida total de validacion para la epoca actual
        val_correct = 0 #Para contar las predicciones correctas
        with torch.no_grad():
            #Se itera sobre los lotes de los datos para evaluar
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device) #Se mueven los datos al dispositivo
                outputs = model(inputs) #Realiza una pasada hacia adelante para obtener las predicciones
                loss = criterion(outputs, labels) #Calcula la perdida de validadcion
                val_loss += loss.item() #Acomula la perdida del lote actual
                #Se cuentan las predicciones correctas en el conjunto de validacion
                val_correct += (outputs.argmax(1) == labels).sum().item()
        val_acc = val_correct / len(val_loader.dataset) #Calcul la precicion de la vaidacion
        print(f"Validación - Pérdida: {val_loss:.4f}, Precisión: {val_acc:.4f}")
        
        
    #--- 4. Guardar el modelo ---
    # Despues de completar el entrenamiento, el modelo se guarda
    torch.save(model.state_dict(), "modelo_resnet_fft.pth")
    print("Modelo guardado en 'modelo_resnet_fft.pth'")

#Punto de entrada para el script principal
if __name__ == "__main__":
    main()
    
    


