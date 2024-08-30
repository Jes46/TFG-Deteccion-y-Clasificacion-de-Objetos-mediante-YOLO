def json2coco(coords_n: list) -> list:
    """
    This function return JSON data in COCO format.
    Format used in COCO "Common Objects in Context" Dataset.
    Format: [x_min, y_min, width_bounding_box, height_bounding_box]
    """
    #coords_n almacena en la primera lista [x_min, y_min] y en la segunda [x_max, y_max]
    x_min = coords_n[0][0]
    y_min = coords_n[0][1]
    x_max = coords_n[1][0]
    y_max = coords_n[1][1]

    box_width = x_max - x_min
    box_height = y_max - y_min

    new_coords = [x_min, y_min, box_width, box_height]
    return new_coords;

def json2pascal_voc(coords_n: list) -> list:
    """
    This function return JSON data in Pascal VOC format.
    Format used in Pascal VOC Dataset. Values are not normalized.
    Format: [x_min, y_min, x_max, y_max]
    """
    #coords_n almacena en la primera lista [x_min, y_min] y en la segunda [x_max, y_max]
    x_min = coords_n[0][0]
    y_min = coords_n[0][1]
    x_max = coords_n[1][0]
    y_max = coords_n[1][1]

    new_coords = [x_min, y_min, x_max, y_max]
    return new_coords;

def json2albumentations(coords_n: list, image_height: int, image_width: int) -> list:
    """
    Similar to Pascal VOC but Albumentations use normalized values.
    Format: [x_min, y_min, x_max, y_max]
    """
    #coords_n almacena en la primera lista [x_min, y_min] y en la segunda [x_max, y_max]
    x_min = coords_n[0][0]
    y_min = coords_n[0][1]
    x_max = coords_n[1][0]
    y_max = coords_n[1][1]

    norm_x_min = x_min / image_width
    norm_y_min = y_min / image_height
    norm_x_max = x_max / image_width
    norm_y_max = y_max / image_height

    #Formato ALBUMENTATIONS:
    new_coords = [norm_x_min, norm_y_min, norm_x_max, norm_y_max]
    return new_coords;

def json2yolo(coords_n: list, image_height: int, image_width: int) -> list:
    """
    This function return JSON data in YOLO format.
    Format usually used in YOLO CNN. Values are normalized.
    Format: [x_center, y_center, width, height] 
    """
    #coords_n almacena en la primera lista [x_min, y_min] y en la segunda [x_max, y_max]
    x_min = coords_n[0][0]
    y_min = coords_n[0][1]
    x_max = coords_n[1][0]
    y_max = coords_n[1][1]
    
    norm_x_center = ((x_max + x_min) / 2) / image_width
    norm_y_center = ((y_max + y_min) / 2) / image_height
    norm_box_width = (x_max - x_min) / image_width
    norm_box_height = (y_max - y_min) / image_height

    #Formato YOLO:
    new_coords = [norm_x_center, norm_y_center, norm_box_width, norm_box_height]
    return new_coords;

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
