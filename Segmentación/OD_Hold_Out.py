import sys
import glob
import time
import yaml
import shutil
import difflib
import JCF_BBoxesM as my
######################################################################################################################################################################################################################################
######################################################################################################################################################################################################################################
start_time = time.time()

time_info = "OD_Hold_Out_MEASUREMENTS.txt"
time_file = open(time_info, 'w')

output_path = 'D:/Datasets/Train-Datasets/ObjectDetection-Dataset/Data_3_(Hold_Out)'
supported_extensions = ['.jpg']

labels_path = glob.glob('D:/Datasets/Train-Datasets/ObjectDetection-Dataset/Augmentations_2/Labels/*.txt')
labels_path.sort(key=len)
images_path = glob.glob('D:/Datasets/Train-Datasets/ObjectDetection-Dataset/Augmentations_2/Images/*.*')
images_path.sort(key=len)

classes_yaml = {1: "Building", 2: "Factory", 3: "Mountains", 4: "Tree", 5: "Wind Turbine"}
dataset_yaml = "D:/Datasets/Train-Datasets/ObjectDetection-Dataset/Data_3_(Hold_Out)/Dataset_3_(Hold_Out).yaml"

train_path = 'D:/Datasets/Train-Datasets/ObjectDetection-Dataset/Data_3_(Hold_Out)/train/images'
val_path = 'D:/Datasets/Train-Datasets/ObjectDetection-Dataset/Data_3_(Hold_Out)/val/images'
with open(dataset_yaml, 'w') as file:
        yaml.safe_dump({
            'train': train_path,
            'val': val_path,
            'names': classes_yaml
        }, file)

def time_measurement(function):
    def func_measured(*args, **kwargs):
        inicio = time.time()
        f = function(*args, **kwargs)
        global time_file
        time_file.write(f"{function} @ {time.time() - inicio} seg.\n")
        return f
    return func_measured

def search_extension(name: str, extensiones: list) -> str:
    alert = True
    for ex in extensiones:
        if(name.endswith(ex) == True):# -> endswith() comprueba si la cadena termina con el sufijo que indiquemos.
            file_ex = ex
            alert = False# -> Extensión del archivo valida.
            return file_ex
        
    if(alert == True):# -> Extensión del archivo no válida.
        pos = name.find('.')
        ex = name[pos:len(name)]
        similitudes = difflib.get_close_matches(ex, extensiones)
        if similitudes:
            print("=======================================================================================================")
            sys.exit(f"\t-> File extension {ex}\n\t-> Extension from {name}{ex} not valid.\n\t-> Suggestion: {similitudes[0]}\n=======================================================================================================")

        else:
            print("=======================================================================================================")
            sys.exit(f"\t-> File extension {ex}\n\t-> Extension from {name}{ex} not valid.\n\t-> No suggestions finded.\n=======================================================================================================")

@time_measurement
def copy_files(image: str, label: str, new_path: str, file_name: str, extensions: list, mode: str):

    if(mode == 'train'):
        # Directorio destino #
        img_to_path = f'{new_path}/train/images/'
        lbl_to_path = f'{new_path}/train/labels/'
        
        image_ex = search_extension(image, extensions)
        shutil.copy(image, f'{img_to_path}{file_name}{image_ex}')
        shutil.copy(label, f'{lbl_to_path}{file_name}.txt')

    elif(mode == 'val'):
        img_to_path = f'{new_path}/val/images/'
        lbl_to_path = f'{new_path}/val/labels/'

        image_ex = search_extension(image, extensions)
        shutil.copy(image, f'{img_to_path}{file_name}{image_ex}')
        shutil.copy(label, f'{lbl_to_path}{file_name}.txt')
    else:
        print("=======================================================================================================")
        sys.exit(f'\t-> Mode {mode} not valid.\n\t-> Only two valid modes: [train, val]')

######################################################################################################################################################################################################################################
######################################################################################################################################################################################################################################

###############################
#      80% Entrenamiento      #
#       20% Validación        #
###############################
p_80 = 1
p_20 = 1
tb = 0
porcentaje = False

for image_file, label_file in zip(images_path, labels_path):
    name = my.search_name(image_file)
    time_file.write("___________________________________\n")
    time_file.write(f"-> {name} <-\n")
    time_file.write("___________________________________\n")

    if((p_80 % 8 != 0) and (porcentaje == False)):
        ######################
        # PATH ENTRENAMIENTO #
        ######################
        copy_files(image_file, label_file, output_path, name, supported_extensions, 'train')

        print(f"\t\t-> Image and Labels from {name} moved to -> Train")

        p_80 += 1
    else:
        porcentaje = True
        p_80 = 1 # Reiniciamos contador.

        if(p_20 < 3):
            ###################
            # PATH VALIDACION #
            ###################
            copy_files(image_file, label_file, output_path, name, supported_extensions, 'val')

            print(f"\t\t-> Image and Labels from {name} moved to -> Val")
        
            p_20 += 1
        else:
            porcentaje = False
            p_20 = 1 # Reiniciamos contador.
            ######################
            # PATH ENTRENAMIENTO #
            ######################
            copy_files(image_file, label_file, output_path, name, supported_extensions, 'train')

            print(f"\t\t-> Image and Labels from {name} moved to -> Train")
    
    
            
end_time = time.time()

time_file.write(f"\n##############################################################################\n")
time_file.write(f">> Tiempo de ejecucion total (CPU): {end_time - start_time} seg.\n")
time_file.write(f"##############################################################################\n")
time_file.close()
