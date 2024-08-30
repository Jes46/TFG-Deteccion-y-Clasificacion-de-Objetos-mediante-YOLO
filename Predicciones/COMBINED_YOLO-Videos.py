import os
import cv2
import numpy as np
from ultralytics import YOLO
import JCF_PredictionsM as my

def draw_menu(image, width: int, height:int, labels_colours: list):
    filledbox_height = 46# -> Altura barra inferior donde se ubicarán las etiquetas.
    filledbox_label = 30# -> Anchura del cuadrado de color para cada etiqueta.
    center = height - filledbox_height/2# -> Centro del rectangulo inferior para introducir las etiquetas de la imagen. Referencia.
    half_height = (filledbox_height/3)# -> El cuadro de color está centrado en la mitad de la barra inferior. Definimos cuanto es de alto desde el medio de la barra para lograr un cuadrado identico por arriba y por abajo. 
    jump = 5# -> Salto que le damos al siguiente cuadrado. Así logramos mantener todos a una distancia similar.
    x_min = 6# -> Punto inicial donde ubicaremos el primer cuadrado de color de las etiquetas.
    
    cv2.rectangle(image, (0, int(height)), (int(width/7), int(height - filledbox_height)), (0, 0, 0), thickness=cv2.FILLED)# -> Barra negra para agregar etiquetas.

    cv2.rectangle(image, (int(x_min), int(center - half_height)), (int(x_min+filledbox_label), int(center + half_height)), labels_colours[0], thickness=cv2.FILLED)# -> Etiqueta BUILDING
    cv2.putText(image, ": Building", (int(x_min+filledbox_label+3), int(center+3)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

    cv2.rectangle(image, (int(jump*filledbox_label), int(center-half_height)), (int(jump*filledbox_label + filledbox_label), int(center+half_height)), labels_colours[1], thickness=cv2.FILLED)# -> Etiqueta FACTORY
    cv2.putText(image, ": Factory", (int(jump*filledbox_label + filledbox_label+3), int(center+3)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)      

    return image
##########################################################################################################################################################################################################################################################################
##########################################################################################################################################################################################################################################################################
object_detection = YOLO('Detection_Hold_Out_FULL.pt')
image_classif = YOLO('Classification_FULL.pt')
DO_model_conf = 0.70
CI_model_conf = 0.80
data = "C:/Users/photo/Desktop/UMA/TFG/Videos/F (2).mp4"
output_path = 'C:/Users/photo/Desktop/UMA/TFG/Code/Entrenamiento&Predicciones/COMBINED_YOLO_Videos/'
name = my.search_name(data)

####################
# CAPTURA DE VIDEO #
####################
cap = cv2.VideoCapture(data)                       # -> Indicar con un 0 en caso de usar cámara externa.
height = int( cap.get(cv2.CAP_PROP_FRAME_HEIGHT) ) # -> Obtenemos la altura del video.
width  = int( cap.get(cv2.CAP_PROP_FRAME_WIDTH) )  # -> Obtenemos la anchura del video.
fps    = cap.get(cv2.CAP_PROP_FPS)                 # -> Obtenemos los FPS del video.
print("====================================================================")
print(f"Video with format: {width}x{height} @ {fps} fps.")
print("====================================================================")

#######################
# INICIO DEL PROGRAMA #
#######################
again = True

while(again == True):
    estado = input("Do you want to use a limit area on the image to do YOLO predictions? Y/N:")

    if(estado == 'y' or estado == "Y"):
        reinicio = True
        total_Buildings = 0
        total_Factorys = 0
        conf_Building_mean = 0
        conf_Factory_mean = 0

        while(reinicio == True):
            ##############################
            # AREA DELIMITADORA ACTIVADA #
            ##############################
            output = f'{output_path}{name}_BoxDetection.mp4'
            coords_interes = list([0, 0, 0, 0])
            coords_interes = my.box_data(coords_interes, width, height)

            record = cv2.VideoWriter(output, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, (width, height))# -> Define el archivo que almacena el video de salida.
            if (cap.isOpened() == False):
                print("##########################")
                print("# Error opening the file #")
                print("##########################")
            else:
                # Se reproduce mientras haya fotorgramas en el video #
                while cap.isOpened():
                    # Leemos fotogramas #
                    success, frame = cap.read()
                    coords = list()   
                    identificadores = list()
                    confianzas = list()
                    
                    # Asignación de Coordenadas Interes #
                    x_min, y_min, x_max, y_max = coords_interes

                    if (success == True):
                        interes = frame[y_min:y_max, x_min:x_max]# -> Recortamos el area de interes del frame original.

                        detection_results = object_detection.predict(interes, conf=DO_model_conf)
                        classification_results = image_classif.predict(interes, conf=CI_model_conf)
                        annotated_frame = classification_results[0].plot()
                        
                        annotated_frame = np.array(annotated_frame)         
                        for result in detection_results:                    
                            for bbox in result.boxes:                       
                                coords.append(bbox.xyxy[0].tolist())        
                                identificadores.append(bbox.cls[0].tolist())
                                confianzas.append(bbox.conf[0].tolist())

                            buildings = identificadores.count(1.0)
                            factorys = identificadores.count(2.0)
                            total_Buildings += buildings
                            total_Factorys += factorys 

                            for ident,confid in zip(identificadores,confianzas):
                                if(ident == 1.0):#Building
                                    conf_Building_mean += confid
                                elif(ident == 2.0):#Factory
                                    conf_Factory_mean += confid

                            for box_coords, id, conf in zip(coords, identificadores, confianzas):
                                x1, y1, x2, y2 = box_coords

                                if(int(id) == 1):# BUILDING
                                    cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
                                    cv2.putText(annotated_frame, f"{round(conf, 2)}", (int(x1+3), int(y1+26)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                                elif(int(id) == 2):# FACTORY
                                    cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (238, 102, 0), 4)
                                    cv2.putText(annotated_frame, f"{round(conf, 2)}", (int(x1+3), int(y1+26)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

                        restricted_height = y_max - y_min
                        restricted_width = x_max - x_min
                        final_frame = my.overlay(restricted_height, restricted_width, frame, annotated_frame, coords_interes)# -> Superposición del frame predicho con el frame original.
                        final_frame = draw_menu(final_frame, width, height, [(0, 255, 0), (238, 102, 0)])# -> Insertamos menu de etiquetas a la imagen final.

                        record.write(cv2.rectangle(final_frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2))
                    else:
                        #####################################################
                        # CIERRE DEL BUCLE SI SE ALCANZA EL FINAL DEL VIDEO #
                        #####################################################
                        break

                ######################################################################
                # VISUALIZACION DEL ULTIMO FRAME PARA VISUALIZAR EL AREA DE INTERES #
                ######################################################################
                cv2.imshow("Restricted Area Display - PRESS 'ESC' TO CLOSE THE WINDOW", final_frame)
                while True:
                    key = cv2.waitKey()
                    if key == 27:# -> Código ascii 'esc'
                        break
                cv2.destroyAllWindows()

                while True:
                    orden = input(">> Does the delimited area correspond to the desired area? Y/N: ")
                    if(orden == 'Y' or orden == 'y'):
                        reinicio = False
                        cap.release()    # -> Cerramos el archivo que lee el video.
                        record.release() # -> Cerramos el archivo que almacena el video.
                        if(total_Buildings != 0):
                            conf_Building_mean = conf_Building_mean / total_Buildings
                        if(total_Factorys != 0):
                            conf_Factory_mean = conf_Factory_mean / total_Factorys
                        print("========================================================================================================================================")
                        print("Video saved in:", output)
                        print(f"Format: {width} x {height} @ {fps} fps.\n\t>> Bounding Box: Enabled")
                        print(f"Total Buildings detected: {total_Buildings}")
                        print(f"Total Factorys detected: {total_Factorys}")
                        print(f"Confidence Building Mean: {conf_Building_mean}")
                        print(f"Confidence Factory Mean: {conf_Factory_mean}")
                        print("========================================================================================================================================")
                        break
                    elif(orden == 'N' or orden == 'n'):
                        reinicio = True
                        cap.release()
                        record.release()
                        os.remove(output)# -> Elimina el video generado con el area delimitadora erronea.
                        total_Buildings = 0
                        total_Factorys = 0
                        conf_Building_mean = 0
                        conf_Factory_mean = 0
                        cap = cv2.VideoCapture(data)
                        break
                    else:
                        print(f"\t -> Character {orden} not valid!")

        again = False# -> Finalizamos bucle una vez confirmemos que el area delimitadora está ubicada en el area deseada.   
    elif(estado == 'n' or estado== "N"):
        total_Buildings = 0
        total_Factorys = 0
        conf_Building_mean = 0
        conf_Factory_mean = 0

        #################################
        # AREA DELIMITADORA DESACTIVADA #
        #################################
        output_path = f'{output_path}{name}_Detection.mp4'
        record = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, (width, height))

        if (cap.isOpened() == False):
            print("##########################")
            print("# Error opening the file #")
            print("##########################")
        else:
            while cap.isOpened():
                success, frame = cap.read() 
                coords = list()   
                identificadores = list()     
                confianzas = list()

                if (success == True):
                    detection_results = object_detection.predict(frame, conf=DO_model_conf)
                    classification_results = image_classif.predict(frame, conf=CI_model_conf)
                    annotated_frame = classification_results[0].plot()
                    
                    annotated_frame = np.array(annotated_frame)         # -> Convertimos la matriz a un formato que permita escritura.
                    for result in detection_results:                    # -> Accedemos a cada objeto Results() que devuelve Predict(). Hay un Results() por cada imagen.
                        for bbox in result.boxes:                       # -> Accedemos a las BBoxes que hay dentro del objeto Results().
                            coords.append(bbox.xyxy[0].tolist())        # -> tolist() función que convierte un Tensor a tipo Lista. Función implementada en Torch -> torch.tensor.tolist -> Documentación: https://pytorch.org/docs/stable/generated/torch.Tensor.tolist.html
                            identificadores.append(bbox.cls[0].tolist())# -> Extraemos los identificadores de los objetos detectados. Formato: [1.0 1.0 1.0 2.0 1.0], por ejemplo.
                            confianzas.append(bbox.conf[0].tolist())    # -> Extraemos las confianzas con las que se ha detectado cada objeto.

                        ###############################
                        # Conteo de clases detectadas #
                        ###############################
                        buildings = identificadores.count(1.0)
                        factorys = identificadores.count(2.0)
                        total_Buildings += buildings
                        total_Factorys += factorys 

                        ##############################################
                        # Calculo de las confianzas medias por clase #
                        ##############################################
                        for ident,confid in zip(identificadores,confianzas):
                            if(ident == 1.0):#Building
                                conf_Building_mean += confid
                            elif(ident == 2.0):#Factory
                                conf_Factory_mean += confid

                        for box_coords, id, conf in zip(coords, identificadores, confianzas):
                            x_min, y_min, x_max, y_max = box_coords

                            if(int(id) == 1):# BUILDING
                                cv2.rectangle(annotated_frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 4)# -> Cada BBox se inserta en el frame con la predicciones del modelo de Clasificación de Imágenes.
                                cv2.putText(annotated_frame, f"{round(conf, 2)}", (int(x_min+3), int(y_min+26)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)# -> Anotamos la confianza con la que se detecto el objeto.   
                            elif(int(id) == 2):# FACTORY
                                cv2.rectangle(annotated_frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (238, 102, 0), 4)
                                cv2.putText(annotated_frame, f"{round(conf, 2)}", (int(x_min+3), int(y_min+26)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                        
                    record.write(draw_menu(annotated_frame, width, height, [(0, 255, 0), (238, 102, 0)]))
                else:
                    break

            cap.release()
            record.release()
            if(total_Buildings != 0):
                conf_Building_mean = conf_Building_mean / total_Buildings
            if(total_Factorys != 0):
                conf_Factory_mean = conf_Factory_mean / total_Factorys
            print("========================================================================================================================================")
            print("Video saved in:", output_path)
            print(f"Format: {width} x {height} @ {fps} fps.\n\t>> Bounding Box: Disabled")
            print(f"Total Buildings detected: {total_Buildings}")
            print(f"Total Factorys detected: {total_Factorys}")
            print(f"Confidence Building Mean: {conf_Building_mean}")
            print(f"Confidence Factory Mean: {conf_Factory_mean}")
            print("========================================================================================================================================")    

        again = False
    else:
        print(f"Character {estado} not valid.")

##########################################################################################################################################################################################################################################################################
##########################################################################################################################################################################################################################################################################

###############
# Anotaciones #
###############

"""
Ver función cv2.createBackgroundSubtractorMOG2 y cv2.createBackgroundSubtractorKNN

Formato que acepta frame es [height, width] donde:
Ejemplo: frame[50:900, 100:1870], la imagen resultante es 1770x850 pixeles
    Ancho = 1870-100 = 1770 pixeles
    Alto = 900-50 = 850 pixeles

    frame[y_min:y_max, x_min:x_max]

    
annotated_frame es una matriz no modificable, declarada como readonly (solo lectura) para el sistema.
Al querer realizar modificaciones sobre ella para insertar las nuevas bboxes se requiere modificar el 
tipo de archivo a una matriz NumPy para poder escribir sobre ella nuevos datos, ¿cómo? utilizando la
funcion:

import numpy as np
annotated_frame = np.array(annotated_frame) -> De este modo se asegura que la matriz es modificable.
                    
"""
