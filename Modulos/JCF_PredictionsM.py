import sys

def search_name(image_path: str) -> str:
    """
    Provides the full name of an image in a string path named <name>
    """
    special = '\ '
    i = 0
    for letter in image_path:
        finded = False
        final = False
        length = len(image_path)       
           
        pos = -1
        while(final == False):
            
            if image_path[pos]=='.':
                pos_1 = length + pos
                finded = True
                
            if finded==True and (image_path[pos]==special[0] or image_path[pos]=='/'):
                pos_0 = length + pos
                final = True
            
            pos -= 1
            
        name = image_path[pos_0+1:pos_1]        
        i += 1
        
        return name

def overlay(height, width, frame, interest, coords: list):
    # Coordenadas que sirven de referencia para ubicar la imagen superpuesta #
    x_min = int( coords[0] )
    y_min = int( coords[1] )
    # Superposición #
    frame[y_min:y_min+height, x_min:x_min+width] = interest

    return frame

def box_data(coords: list, width: int, height: int) -> list:
    print("Add the coords of the Bounding Box below:")
    coords[0] = int( input(">> x_min:") )
    coords[2] = int( input(">> x_max:") )
    coords[1] = int( input(">> y_min:") )
    coords[3] = int( input(">> y_max:") )
    # Convertimos a enteros para que sean copmpatibles los tipos de datos con la librería cv2
    if((coords[2]-coords[0] > width) or (coords[3]-coords[1] > height)):
        print("=======================================================================================================")
        sys.exit(f"\t-> Box size not valid!\n\t-> Box size: {coords[2]-coords[0]}x{coords[3]-coords[1]}\n\t-> Image size: {width}x{height}\n=======================================================================================================")
    else:
        return coords
