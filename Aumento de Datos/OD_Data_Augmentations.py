import sys
import cv2
import time
import yaml
import glob
import json
import difflib
import JCF_BBoxesM as my
from pathlib import Path
import albumentations as A
"""
Special mention to Albumentations authors:
@Article{info11020125,
    AUTHOR = {Buslaev, Alexander and Iglovikov, Vladimir I. and Khvedchenya, Eugene and Parinov, Alex and Druzhinin, Mikhail and Kalinin, Alexandr A.},
    TITLE = {Albumentations: Fast and Flexible Image Augmentations},
    JOURNAL = {Information},
    VOLUME = {11},
    YEAR = {2020},
    NUMBER = {2},
    ARTICLE-NUMBER = {125},
    URL = {https://www.mdpi.com/2078-2489/11/2/125},
    ISSN = {2078-2489},
    DOI = {10.3390/info11020125}
}
"""
######################################################################################################################################################################################################################################
######################################################################################################################################################################################################################################
cpu_start_time = time.time()

time_info = "OD_Data_Augmentations_MEASUREMENTS.txt"
time_file = open(time_info, 'w')
labels_list = list(["Building", "Factory", "Mountains", "Tree", "Wind Turbine"])# -> Escribir etiquetas en el mismo orden según el identificador que tengan asociado.

datasetet_path = Path('D:/Datasets/OD-Buildings-Factory-Dataset')

output_images_path = 'D:/Datasets/Train-Datasets/ObjectDetection-Dataset/Augmentations_2/Images/'
output_labels_path = 'D:/Datasets/Train-Datasets/ObjectDetection-Dataset/Augmentations_2/Labels/'

dataset_info = "D:/Datasets/Train-Datasets/ObjectDetection-Dataset/Augmentations_2/Dataset_Info.txt"
dataset_yaml = "D:/Datasets/Train-Datasets/ObjectDetection-Dataset/Augmentations_2/Augmentations.yaml"
classes_yaml = {1: "Building", 2: "Factory", 3: "Others", 4: "Others", 5: "Others"}# -> Indicamos en un diccionario ID:Etiqueta -> Clave:Valor del diccionario

im_supported_extensions = ['.jpg']
coords_supported_extensions = ['.json']
def time_measurement(function):
    def func_measured(*args, **kwargs):
        inicio = time.time()
        f = function(*args, **kwargs)
        global time_file
        time_file.write(f"{function} @ {time.time() - inicio} seg.\n")
        return f
    return func_measured

def labels_filter(label: str, label_id: int, name: str, my_list: list):
    """
    Function used to filter identifiers and labels. This ensures that no erroneous labels or 
    incorrectly assigned identifiers enter as data.
    """
    try:
        if(my_list.index(label) != ValueError):
            
            if(my_list.index(label) != label_id-1):# -> Se resta 1 porque en este TFG el identificador mas pequeño es 1: Building. No hay etiqueta asignada al identificador 0.
                print("=======================================================================================================")
                sys.exit(f"\t-> Error en el archivo: {name}.json\n\t-> Identificador ({label_id}) no valido para la etiqueta <{label}>\n\t-> Identificador esperado: {my_list.index(label)+1}\n=======================================================================================================")
    
    except ValueError:
        # No se encuentra en la lista la etiqueta especificada. Procedemos a evaluar si encontramos alguna similar.
        similitudes = difflib.get_close_matches(label, my_list)
        if similitudes:
            print("=======================================================================================================")
            sys.exit(f"\t-> Error en el archivo: {name}.json\n\t-> Etiqueta <{label}> no válida\n\t-> Sugerencia: {similitudes[0]}\n=======================================================================================================")

        else:
            print("=======================================================================================================")
            sys.exit(f"\t-> Error en el archivo: {name}.json\n\t-> Etiqueta <{label}> no valida\n\t-> No se encontraron similitudes válidas.\n=======================================================================================================")

def YOLO_writer_txt(path: str, file_name: str, bboxes: list, image_ids_list: list):
    """
    This function need the absolute path to the directory where we are saving the file:
        path = C:/Users/.........../
    And file_name wich is the name we want to name the file:
        name = Building
    Them the function link both string:
        C:/Users/.........../Building -> Also the function add the file type .txt at the end.
    Finally we have:
        C:/Users/.........../Building.txt

    Writing format: id_label_bbox coords_bbox -> Example: 0 0.2 0.689 0.45 0.782
    """
    txt_file = open(f"{path}{file_name}.txt", 'w')# -> Creamos archivo donde se guardan las BBoxes en el formato necesario para usar en la CNN.
    id_cont = 0# -> Nos desplaza por la lista de identificadores de la imagen.

    for coords_n in bboxes:# -> coords_n es una lista que contiene las coordenadas de cada BBox.
        txt_file.write(str(image_ids_list[id_cont]))# -> Escribimos el id correspondiente a la BBox.
        txt_file.write(' ')

        for num in coords_n:# -> Escribimos las coordenadas una a una de la BBox.
            txt_file.write(str(num))
            txt_file.write(' ')   
        txt_file.write('\n')# -> Salto de linea para la siguiente BBox.

        id_cont += 1
    txt_file.close()# -> Cerramos el flujo de escritura y así guardar correctamente el archivo.

def save_data(trans_image, image_path:str, image_name:str, label_path:str, bboxes: list, image_ids: list, index: int):
    
    new_image = cv2.cvtColor(trans_image, cv2.COLOR_RGB2BGR)# -> Modificamos nuevamente el espacio de color para guardar la imagen correctamente con cv2.
    cv2.imwrite(image_path + image_name + f"_{index}.jpg", new_image)# -> Guardamos la primera imagen generada.
    YOLO_writer_txt(path = label_path, file_name = image_name + f'_{index}', bboxes = bboxes, image_ids_list = image_ids)# Guardamos nuevas coordenadas de la imagen generada.

######################################################################################################################################################################################################################################
######################################################################################################################################################################################################################################
start = time.time()

images_list = list()
json_list = list()
for ext in im_supported_extensions:
    # Bucle que itera todo el path para quedarse con los archivos con los formatos compatibles #
    images_list.extend(sorted((datasetet_path).rglob(f"*{ext}")))
for ext in coords_supported_extensions:
    json_list.extend(sorted((datasetet_path).rglob(f"*{coords_supported_extensions}")))

end = time.time()
time_file.write(f"> Tiempo de filtrado de imagenes y archivos json @ {end - start} seg.\n")

if(len(images_list) != len(json_list)):
    print("############################################################################################################")
    print(f"There are not the same number of images and json files.\n>> Json files: {len(json_list)}\n>> Images files: {len(images_list)}")
    sys.exit("############################################################################################################")

with open(dataset_yaml, 'w') as file:
        yaml.safe_dump({
            'names': classes_yaml,
        }, file)
######################################################################################################################################################################################################################################
######################################################################################################################################################################################################################################
tb = 0.0
num_Building = 0
num_Factory = 0
num_Mountains = 0
num_Trees = 0
num_Wind_Turbines = 0
tiempo_total_lectura_jsons = 0
tiempo_transformadas_total = 0
tiempo_trans1_total = 0
tiempo_trans2_total = 0


for json_file, image_file in zip(json_list, images_list):
    start_it_n = time.time()
    json_file = str(json_file)
    json_name = my.search_name(json_file)

    time_file.write("___________________________________\n")
    time_file.write(f"-> {json_name} <-\n")
    time_file.write("___________________________________\n")

    print("-> ", json_name, "\n\t-> Processing JSON file...")

    start_json = time.time()
    
    with open(json_file, 'r') as file:
        json_data = json.load(file)
        imageHeight = json_data["imageHeight"]# -> Altura de la imagen
        imageWidth  = json_data["imageWidth"] # -> Anchura de la imagen
        metadata    = json_data["shapes"]     # -> Información sobre "shapes" en el apartado "ANOTACIONES" al final de este documento.

    image_ids          = list()# -> Lista con los identificadores de cada etiqueta de una imagen.    
    new_bboxes         = list()# -> BBoxes formateadas del archivo .json al nuevo formato.
    image_labels       = list()# -> Lista con las etiquetas que contiene una imagen.
    image_ids_to_name  = dict()# -> Diccionario con los identificadores junto a su etiqueta de cada imagen. Formato: image_ids_to_name = {0: "Building", 1: "Desert"} -> Necesario para visualizar ls BBoxes con las funciones de visualización.

    for values in metadata:# -> Se repite tantas veces como BBoxes contenga la imagen.
        label        = values["label"]   # -> Guardamos etiqueta de la bbox.
        group_id     = values["group_id"]# -> Guardamos id de la bbox.
        bounding_box = values["points"]  # -> Guardamos coordenadas de la bbox.

        if(group_id == None):
            print("=======================================================================================================")
            sys.exit(f"\t-> Error de identificador. Archivo <{json_name}.json> sin identificador de etiqueta.\n=======================================================================================================")
        
        labels_filter(label, group_id, json_name, labels_list)
        if(label == labels_list[0]):
            num_Building += 1
        elif(label == labels_list[1]):
            num_Factory += 1
        elif(label == labels_list[2]):
            num_Mountains += 1
        elif(label == labels_list[3]):
            num_Trees += 1
        elif(label == labels_list[4]):
            num_Wind_Turbines += 1

        #####################################
        # ALMACENAMIENTO DATOS ARCHIVO JSON #
        #####################################
        image_ids.append(group_id)# -> Agregamos a la lista los identificadores.
        image_labels.append(label)# -> Agregamos a la lista las etiquetas.
        image_ids_to_name[group_id] = label# -> Agregamos Clave-Valor al Diccionario.
        new_bboxes.append(my.json2yolo(bounding_box, image_height=imageHeight, image_width=imageWidth))# -> Agregamos a la lista las nuevas coordenadas en el formato indicado.
    
    end_json = time.time()
    tiempo_total_lectura_jsons = tiempo_total_lectura_jsons + (end_json - start_json)

    ###################################
    # DEFINICIÓN DE LAS TRANSFORMADAS #
    ###################################
    data_transforms_1 = A.Compose([
        A.Rotate(limit=[-8, 8], interpolation=cv2.INTER_LINEAR, p=1),  # -> Rotación aleatoria entre -10 y 10 grados.
        A.ISONoise(intensity=(0.3,0.5), p=0.5)                         # -> Ruido sensor cámara.
    ], bbox_params=A.BboxParams(format='yolo', label_fields=["class_labels", "category_ids"]))

    data_transforms_2 = A.Compose([
        A.MotionBlur(blur_limit=(3, 33), p=0.8),                                        # -> Desenfoque de movimiento.
        A.HueSaturationValue(hue_shift_limit=(-5, 5), sat_shift_limit=(-5,5) , p=1)     # -> Variación de la luminosidad, tonos y saturación.
    ], bbox_params=A.BboxParams(format='yolo', label_fields=["class_labels", "category_ids"]))
    
    # Abrimos imagen para trabajar conjuntamente con ella y sus BBoxes #
    image_file = str(image_file)
    image_name = my.search_name(image_file)
    if(image_name != json_name):
        print("Json file name:", json_name)
        print("Image name:", image_name)
        sys.exit("\t>> ALERT! Json and Image name are not the same.\n\t>> Fatal Error!")

    image = cv2.imread(image_file)
    print(f"\t-> Processing augmentations to {image_name}...")

    cv2.imwrite(output_images_path + image_name + ".jpg", image)# -> Guardamos original en nuevo directorio.
    YOLO_writer_txt(path=output_labels_path, file_name=image_name, bboxes=new_bboxes, image_ids_list=image_ids)# -> Guardamos coordenadas en un .txt

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)# -> Funcion empleada para cambiar el espacio de color de la imagen. Se hace porque CV2 trabaja en BGR y Albumentations en RGB.

    trans_start_time = time.time()

    tiempo_trans_1_start = time.time()

    transformed_1 = data_transforms_1(image=image, bboxes=new_bboxes, class_labels=image_labels, category_ids=image_ids)# -> Ejecutamos el canal de aumento 1.
    save_data(transformed_1["image"], output_images_path, image_name, output_labels_path, transformed_1["bboxes"], transformed_1["category_ids"], 1)# -> Guardamos imagen generada.
    
    tiempo_trans_1_end = time.time()
    tiempo_trans1_total = tiempo_trans1_total + (tiempo_trans_1_end - tiempo_trans_1_start)

    tiempo_trans_2_start = time.time()

    transformed_2 = data_transforms_2(image=image, bboxes=new_bboxes, class_labels=image_labels, category_ids=image_ids)
    save_data(transformed_2["image"], output_images_path, image_name, output_labels_path, transformed_2["bboxes"], transformed_2["category_ids"], 2)
    
    tiempo_trans_2_end = time.time()
    tiempo_trans2_total = tiempo_trans2_total + (tiempo_trans_2_end - tiempo_trans_2_start)

    trans_end_time = time.time()
    tiempo_transformadas_total = tiempo_transformadas_total + (trans_end_time - trans_start_time)

    print(f"\t\t-> Augmentations to {image_name} finished.")

    end_it_n = time.time()
    ta = end_it_n - start_it_n
    time_file.write(f"> Tiempo ejecucion {json_name} @ {ta} seg.\n")
    if(ta > tb):
        max_it = json_name
        max_time = ta
        tb = ta
    else:
        tb = ta


############
#   MENU   #
############
print("-------------------------------------------------------------------")
print("####################")
print("# Original Dataset #")
print("####################")
print("> Images:", len(images_list))
print("> JSON files:", len(json_list))
print("###############")
print("# New Dataset #")
print("###############")
print("> Images:", len( glob.glob(output_images_path + '*.*') ))
print("> TXT files:", len( glob.glob(output_labels_path +'*.txt') ))

txt = open(dataset_info, 'w')
txt.write("####################\n")
txt.write("# Original Dataset #\n")
txt.write("####################\n")
txt.write(f"> Images: {len(images_list)}\n")
txt.write(f"> JSON files: {len(json_list)}\n")
txt.write("###############\n")
txt.write("# New Dataset #\n")
txt.write("###############\n")
txt.write(f"> Images: {len( glob.glob(output_images_path + '*.*') )}\n")
txt.write(f"> TXT files: {len( glob.glob(output_labels_path +'*.txt') )}\n")
txt.close()
cpu_end_time = time.time()

time_file.write(f"\n##############################################################################\n")
time_file.write(f">> Tiempo de ejecucion total (CPU): {cpu_end_time - cpu_start_time} seg.\n")
time_file.write(f">> Iteracion con mayor tiempo de procesado: {max_it} @ {max_time} seg.\n")
time_file.write(f">> Tiempo total procesado archivos Json: {tiempo_total_lectura_jsons} seg.\n")
time_file.write(f">> Tiempo total procesado Transformadas: {tiempo_transformadas_total} seg.\n")
time_file.write(f">> Tiempo total procesado Transformada 1: {tiempo_trans1_total} seg.\n")
time_file.write(f">> Tiempo total procesado Transformada 2: {tiempo_trans2_total} seg.\n")
time_file.write(f">> Instancias Building: {num_Building}\n")
time_file.write(f">> Instancias Factory: {num_Factory}\n")
time_file.write(f">> Instancias Montañas: {num_Mountains}\n")
time_file.write(f">> Instancias Arboles: {num_Trees}\n")
time_file.write(f">> Instancias Turbinas de viento: {num_Wind_Turbines}\n")
time_file.write(f"##############################################################################\n")
time_file.close()
