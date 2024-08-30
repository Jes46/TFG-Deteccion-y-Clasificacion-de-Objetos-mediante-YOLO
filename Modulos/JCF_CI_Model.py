import random

def rand() -> float:
    """
    Create a random number between [-25, 25] to rotate the image
    no more than 25 degrees to the left or 25 degrees to the right.
    """
    rotation = round( random.uniform(-25, 25), 1)
    return rotation;
    
def rand_jitter() -> float:
    """
    Special function to use with augly.image.color_jitter() function.
    The values into the interval of random.uniform() function were previously
    studied individually with each factor in .color_jitter() function
    to know how they worked and get logical results. 
    """
    rotation = round(random.uniform(0.46, 1.20), 1)
    return rotation;

def search_number(path: str) -> int: 
    """
    This function looks for the special sequence (<number>), example: (46)
    With this we are forcing using a special naming in our Data Base Images.
    
    The nomenclature we want is:
        <file_name>(<number>).<format>, example: Buildings(46).jpg
        
    Then, this function return the number between '(' and ')'
    """
    n = 0
    booleano = False
    finded = False
        
    for letter in path:
    
        if letter == '(' and finded == False:
            pos_0 = n
            booleano = True
        
        if booleano == True and letter== ')':
            pos_1 = n    
            booleano = False
            
            if len( path[pos_0+1:pos_1] ) != 0:
                finded = True
            
        n += 1
        
    num = path[pos_0+1:pos_1]
    return num

def search_name(image_path: str ) -> str:
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
