import os
import cv2
import sys
import time
import glob
import errno
import JCF_CI_Model as my
from openpyxl import Workbook                               # -> Libreria para trabajar con archivos Excel.
import augly.image as auimage
######################################################################################################################################################################################################################################
######################################################################################################################################################################################################################################
cpu_start_time = time.time()
image_path = glob.glob('D:/Datasets/OC-Image-Dataset/*.jpg')
image_path.sort(key=len)# -> Ordenamos de  menor a mayor según la longitud del string.

excel_path = 'D:/Datasets/Train-Datasets/ImageClassification-Dataset/Data_2/'
train_path = 'D:/Datasets/Train-Datasets/ImageClassification-Dataset/Data_2/train/'
val_path = 'D:/Datasets/Train-Datasets/ImageClassification-Dataset/Data_2/val/'
dataset_info = 'D:/Datasets/Train-Datasets/ImageClassification-Dataset/Data_2/Dataset_Info.txt'

excel_file = '(0)Etiquetado.xlsx'# -> Archivo excel donde guardamos el nombre de la imagen junto a su correspondiente indicador. (0: No edificio, 1:Edificio)
time_info = "OC_Data_Augmentations_MEASUREMENTS.txt"
time_file = open(time_info, 'w')
objects_list = list(["Buildings", "Factory", "Countryside", "Mountains"])
######################################################################################################################################################################################################################################
######################################################################################################################################################################################################################################
book = Workbook()                                           # -> Inicializamos la variable que será finalmente nuestro archivo Excel.
etiquetado = book.active                                    # -> Variable que indica la pagina que activamos en el Excel donde vamos a escribir. Por defecto es la primera hoja.
celda = 1                                                   # -> Declaramos variable global para enumerar la sucesion de celdas donde almacenamos los nombres de las imagenes.

aug_per_train_image = 2 # -> Variable informativa
aug_per_val_image = 2   # -> Variable informativa


def is_building(path: str) -> bool:
    """
    Funcion que escribe en el archivo excel el nombre de la imagen original e indica con 0: NO o 1: SI, si en ella se encuentra
    el objeto de interes <interest_object>.
    """
    global celda
    building = False
    my_objects_list = path.split()

    interest_object = "Buildings" 
    if interest_object in my_objects_list:
        etiquetado[f'A{celda}'] = path
        etiquetado[f'B{celda}'] = 1
        celda += 1
        building = True
    else:
        etiquetado[f'A{celda}'] = path
        etiquetado[f'B{celda}'] = 0
        celda += 1
        
    return building;                                        # -> Variable externa que sirve para reducir costes computacionales que indica si las siguientes imagenes generadas a la original contienen edificios o no.

def write_format(image: str, index: int, booleano: bool):
    global celda
    if booleano == True:                                    # -> Una vez conocido el contenido de la imagen original, con el booleano indicamos a las modificadas que objeto contienen.
        etiquetado[f'A{celda}'] = image + '_' + str(index)
        etiquetado[f'B{celda}'] = 1                         # -> Imagen generada contiene edificio.
        celda += 1
    else:
        etiquetado[f'A{celda}'] = image + '_' + str(index)
        etiquetado[f'B{celda}'] = 0                         # -> Imagen generada no contiene edificio.
        celda += 1  

def train_code(image_n: str, objects_list: list, train_path: str, contador: int) -> int:
    name = my.search_name(image_n)
    objects = name.split()#-> Convertir a lista el string. Si hay espacios, creará una lista de listas.
    objects.pop()# -> Elimina la ultima lista que contiene el número de la imagen.
    
    for label in objects:# -> Por si el nombre de la imagen incluye mas de una etiqueta separada con espacios.
        if label in objects_list:
            try:
                save_path = f'{train_path}{label}' 
                os.makedirs(save_path, exist_ok=True) # -> Creamos directorio.
            except OSError as e:# -> Error que se genera si el directorio ya existe. Se ignora para que no cierre la ejecución del programa.
                if e.errno != errno.EEXIST:
                    raise
                
            print("-> ", name, "\n\t-> [Train]-Processing...")

            image = cv2.imread(image_n)
            cv2.imwrite(save_path + '/' + name + ".jpg", image)# -> Guardamos original en el nuevo directorio.    
            content = is_building(name)
            
            output_path = save_path + '/' + name + '_1.jpg'
            auimage.blur(image_n, output_path=output_path, radius=4)# Desenfoque
            write_format(name, index=1, booleano=content)     
            contador += 1
            
            output_path = save_path + '/' + name + '_2.jpg'
            auimage.color_jitter(image_n, output_path=output_path, brightness_factor=my.rand_jitter(),
                                    contrast_factor=my.rand_jitter(), saturation_factor=my.rand_jitter())
            write_format(name, index=2, booleano=content)# Ajuste aleatorio de Brillo, Contraste y Saturación.  
            contador += 1
            
            print("\t\t-> Finished.")
        else:
            print("=======================================================================================================")
            sys.exit(f"\t-> Error en la imagen <{name}.jpg>.\n\tEtiqueta {label} no valida.\n=======================================================================================================")

    return contador

def test_code(image_n: str, objects_list: list, val_path: str, contador: int) -> int:
    name = my.search_name(image_n)
    objects = name.split()
    objects.pop()
    
    
    for label in objects:
        if label in objects_list:
            try:
                save_path = f'{val_path}{label}' 
                os.makedirs(save_path, exist_ok=True) # -> Creamos directorio.
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise
                      
            print("-> ", name, "\n\t-> [Val]-Processing...")

            image = cv2.imread(image_n)
            cv2.imwrite(save_path + '/' + name + ".jpg", image) 
            content = is_building(name)  
                        
            output_path = save_path + '/' + name + '_1.jpg'
            auimage.blur(image_n, output_path=output_path, radius=4)
            write_format(name, index=1, booleano=content)     
            contador += 1
            
            output_path = save_path + '/' + name + '_2.jpg'
            auimage.color_jitter(image_n, output_path=output_path, brightness_factor=my.rand_jitter(),
                                    contrast_factor=my.rand_jitter(), saturation_factor=my.rand_jitter())
            write_format(name, index=2, booleano=content)
            contador += 1
            
            print("\t\t-> Finished.")
        else:
            print("=======================================================================================================")
            sys.exit(f"\t-> Error en la imagen <{name}.jpg>.\n\tEtiqueta {label} no valida.\n=======================================================================================================")

    return contador

######################################################################################################################################################################################################################################
######################################################################################################################################################################################################################################

###############################
#      80% Entrenamiento      #
#       20% Validación        #
###############################
p_80 = 1
p_20 = 1
tb = 0.0
porcentaje = False
tiempo_total_train = 0
tiempo_total_val = 0

train_aug = 0
val_aug = 0
for image_n in image_path:
    start_it_n = time.time()
    image_name = my.search_name(image_n)
    time_file.write("_______________________________________\n")
    time_file.write(f"-> {image_name} <-\n")
    time_file.write("_______________________________________\n\n")
    
    if((p_80 % 8 != 0) and (porcentaje == False)):
        ######################
        # PATH ENTRENAMIENTO #
        ######################
        comienzo = time.time()

        train_aug = train_code(image_n, objects_list, train_path, train_aug)

        final = time.time()
        tiempo_total_train = tiempo_total_train + (final - comienzo)

        p_80 += 1
    else:
        porcentaje = True
        p_80 = 1# -> Se reinicia el contador.

        if(p_20 < 3):
            ###################
            # PATH VALIDACION #
            ###################
            comienzo = time.time()

            val_aug = test_code(image_n, objects_list, val_path, val_aug)

            final = time.time()
            tiempo_total_val = tiempo_total_val + (final - comienzo)

            p_20 += 1
        else:
            ######################
            # PATH ENTRENAMIENTO #
            ######################
            comienzo = time.time()

            train_aug = train_code(image_n, objects_list, train_path, train_aug)

            final = time.time()
            tiempo_total_train = tiempo_total_train + (final - comienzo)

            porcentaje = False
            p_20 = 1# -> Se reinicia el contador.
    
    end_it_n = time.time()
    ta = end_it_n - start_it_n
    if(ta > tb):
        max_it = image_name
        max_time = ta
        tb = ta
    else:
        tb = ta
        
cpu_end_time = time.time()
time_file.write(f"\n##############################################################################\n")
time_file.write(f">> Tiempo de ejecucion total (CPU): {cpu_end_time - cpu_start_time} seg.\n")
time_file.write(f">> Iteracion con mayor tiempo de procesado: {max_it} @ {max_time} seg.\n")
time_file.write(f">> Tiempo total train_code: {tiempo_total_train} seg.\n")
time_file.write(f">> Tiempo total val_code: {tiempo_total_val} seg.\n")
time_file.write(f"##############################################################################\n")     
book.save(excel_path + excel_file)# -> Se guarda el archivo como documento externo.

######################################################################################################################################################################################################################################
######################################################################################################################################################################################################################################
########
# MENU #
########
print("-------------------------------------------------------------------")
print("####################")
print("# Original Dataset #")
print("####################")
print("> Total images:", len(image_path))
print("> Augmentations per train image: ", str(aug_per_train_image))
print("> Augmentations per val image: ", str(aug_per_val_image))
print("-------------------------------------------------------------------")
print("###############")
print("# New Dataset #")
print("###############")
print("> Total train folders created:", len(objects_list))
print("> Total val folders created:", len(objects_list))
print(f"> All lavels saved in {excel_file}")
print(">> NEW TRAIN FOLDERS:")
pos = 0
for label in objects_list:
    path = glob.glob(f'{train_path}{objects_list[pos]}/*.jpg')
    print(f'\t-> {train_path}{objects_list[pos]} with {len(path)} images.')
    pos += 1
print(">> NEW VAL FOLDERS:")
pos = 0
for label in objects_list:
    path = glob.glob(f'{val_path}{objects_list[pos]}/*.jpg')
    print(f'\t-> {val_path}{objects_list[pos]} with {len(path)} images.')
    pos += 1

txt = open(dataset_info, 'w')
txt.write("-------------------------------------------------------------------\n")
txt.write("####################\n")
txt.write("# Original Dataset #\n")
txt.write("####################\n")
txt.write(f"> Total images: {len(image_path)}\n")
txt.write(f"> Augmentations per train image: {str(aug_per_train_image)}\n")
txt.write(f"> Augmentations per val image: {str(aug_per_val_image)}\n")
txt.write("-------------------------------------------------------------------\n")
txt.write("###############\n")
txt.write("# New Dataset #\n")
txt.write("###############\n")
txt.write(f"> Total train folders created: {len(objects_list)}\n")
txt.write(f"> Total val folders created: {len(objects_list)}\n")
txt.write(f"> All lavels saved in {excel_file}\n")
txt.write(">> NEW TRAIN FOLDERS:\n")
pos = 0
for label in objects_list:
    path = glob.glob(f'{train_path}{objects_list[pos]}/*.jpg')
    txt.write(f'\t-> {train_path}{objects_list[pos]} with {len(path)} images.\n')
    pos += 1
txt.write(">> NEW VAL FOLDERS:\n")
pos = 0
for label in objects_list:
    path = glob.glob(f'{val_path}{objects_list[pos]}/*.jpg')
    txt.write(f'\t-> {val_path}{objects_list[pos]} with {len(path)} images.\n')
    pos += 1
txt.close()

time_file.close()

######################################################################################################################################################################################################################################
######################################################################################################################################################################################################################################
###############
# ANOTACIONES #
###############
"""
image_path = 'C:/Users/photo/Desktop/UMA/TFG/Pruebas-Data-Base/Buildings(1).jpg' -> Es un String

image_path = glob.glob('C:/Users/photo/Desktop/UMA/TFG/Pruebas-Data-Base/*.jpg') -> Lista con todos los path de las imagenes con formato .jpg de ese directorio.


Para abrir todas las imágenes de formatos distintos se debe indicar del 
siguiente modo:
    -> 'C:/Users/photo/Desktop/UMA/TFG/Pruebas-Data-Base/*.*'
    Importante que solo existan en el directorio imágenes sino, abrira archivos
    indeseados y generará errores de ejecución de nuestro código.
Además, hace distinción entre los formatos JPG y JPEG por lo que se deberá de tener
cuidado con el formato con el que guardamos las imágenes para evitar errores.



try: Inicia un bloque de código en el que se intenta ejecutar ciertas operaciones.

os.mkdir('dir1'): Intenta crear un directorio llamado 'dir1' utilizando la función os.mkdir(). Esta función pertenece al módulo os y se 
utiliza para crear un directorio con el nombre especificado.

except OSError as e: Si ocurre una excepción de tipo OSError, se captura y se almacena en la variable e.

if e.errno != errno.EEXIST: Verifica si el número de error (errno) en la excepción e no es igual al código 
de error EEXIST (que significa que el directorio ya existe). EEXIST es un código de error específico que indica que el archivo o 
directorio ya existe.

raise: Si la excepción no es debida a que el directorio ya existe, entonces se vuelve a lanzar la excepción. 
Esto significa que cualquier otro tipo de error que no sea la existencia previa del directorio se propaga y no es 
manejado explícitamente en este bloque.

En resumen, el código intenta crear el directorio 'dir1', y si ya existe, simplemente pasa sin generar un error. 
Si surge algún otro error durante la creación del directorio, se propaga y no se maneja en este bloque específico, 
lo cual podría ser capturado por un bloque except superior si lo hubiera en el código circundante.
"""
