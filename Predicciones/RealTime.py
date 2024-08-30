import cv2
import numpy as np
from ultralytics import YOLO
#############################################################
#############################################################

# DELCARACÓN DE MODELOS YOLO Y CONFIANZAS #
object_detection = YOLO('Detection_Hold_Out_FULL.pt')
image_classif = YOLO('Classification_FULL.pt')
DO_model_conf = 0.70
CI_model_conf = (0.15*DO_model_conf) + DO_model_conf

# CONFIGURACION ARCHIVO DE SALIDA #
output_path = 'C:/Users/photo/Desktop/UMA/TFG/Code/Entrenamiento&Predicciones/COMBINED_YOLO_Videos/Real_Time/'
video_name = input(">> Insert the file name: ")
output = f"{output_path}{video_name}.avi"

# APERTURA DE LA CAMARA Y TOMA DE DATOS DE CAPTURA #
cap = cv2.VideoCapture(0)# Abre la primera cámara conectada !
height = int( cap.get(cv2.CAP_PROP_FRAME_HEIGHT) ) # -> Obtenemos la altura del video.
width  = int( cap.get(cv2.CAP_PROP_FRAME_WIDTH) )  # -> Obtenemos la anchura del video.
#fps    = cap.get(cv2.CAP_PROP_FPS)                 # -> Obtenemos los FPS del video.
fps = 25 # -> Se fuerzan a 25 fps. Necesidades del sistema.


fourcc = cv2.VideoWriter_fourcc(*'XVID')
RT_record = cv2.VideoWriter(output, fourcc, fps, (width, height))


if not cap.isOpened():
    print("#########################")
    print("Error opening the camera!")
    print("#########################")

print("#################")
print("> System ready...")
print("#################")

while True:
    
    estado, frame = cap.read()
    coords = list()   
    identificadores = list()
    confianzas = list()

    if (estado==False):
        print("##########################")
        print("> Error reading the frame.")
        print("\t> System closed.")
        print("##########################")
        break

    # PREDICCIONES DE LOS MODELOS #
    detection_results = object_detection.predict(frame, conf=DO_model_conf)
    classification_results = image_classif.predict(frame, conf=CI_model_conf)
    annotated_frame = classification_results[0].plot()
    annotated_frame = np.array(annotated_frame)     

    # EXTRACCIÓN DE DATOS DEL MODELO detection_results #
    for result in detection_results:                    
        for bbox in result.boxes:                       
            coords.append(bbox.xyxy[0].tolist())        
            identificadores.append(bbox.cls[0].tolist())
            confianzas.append(bbox.conf[0].tolist())
        
        # MOSTRAR EN EL FRAME DEL MODELO classification_results LOS RESULTADOS DEL MODELO detection_results #
        for box_coords, id, conf in zip(coords, identificadores, confianzas):
            x1, y1, x2, y2 = box_coords

            if(int(id) == 1):# BUILDING (VERDE)
                cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)
                cv2.putText(annotated_frame, f"{round(conf, 2)}", (int(x1+3), int(y1+26)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1)
            elif(int(id) == 2):# FACTORY (AZUL)
                cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (238, 102, 0), 1)
                cv2.putText(annotated_frame, f"{round(conf, 2)}", (int(x1+3), int(y1+26)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1)

    
    # GRABACIÓN DE PREDICCIONES Y VISUALIZACÓN POR PANTALLA#

    RT_record.write(annotated_frame)
    cv2.imshow('Real Time', annotated_frame)
    

    # Si se pulsa ESC fin del programa (ESC tiene el código ASCII 27)
    if cv2.waitKey(1) & 0xFF == 27:
        print("Press ESC to close...")
        break



cap.release()# Se libera la captura de la webcam.
RT_record.release()# Se libera la cpatura de video.
cv2.destroyAllWindows()# Se detruyen todas las ventanas.
print("> Closing system...")
print(f"Video Format @ {fps} fps || {width}x{height}")



"""
ANOTACIONES
___________

! if cv2.waitKey(1) & 0xFF == 27:

    (1) cv2.waitKey(1): Esta función espera una tecla por un tiempo específico en milisegundos (en este caso, 1 milisegundo). Si una tecla es presionada durante este tiempo, 
        devuelve el código ASCII de la tecla presionada. Si no se presiona ninguna tecla, devuelve -1.

    (2) & 0xFF: El operador & es un operador bit a bit. 0xFF es el valor hexadecimal para 255, que en binario es 11111111. Al hacer cv2.waitKey(1) & 0xFF, 
        estamos enmascarando el valor devuelto por cv2.waitKey(1) para quedarnos solo con los últimos 8 bits del valor. Esto se hace porque cv2.waitKey(1) puede 
        devolver un valor de 32 bits, pero los códigos ASCII relevantes para las teclas son de 8 bits. De esta manera, nos aseguramos de obtener un valor en el rango correcto.

    (3) == 27: El 27 es el código ASCII para la tecla "Esc". Entonces, esta comparación verifica si la tecla presionada es la tecla "Esc".


"""
