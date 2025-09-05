import numpy as np

def dotting(f: list, other_info = None) -> tuple:
    '''
    Given a function, return 'dotted' version
    
    Args:
        f: function to be dotted
            - first argument is x value
            - functions are formatted as:
                (f, a, b)
        other_info: any other information needed to compute the dotted version
        
        
    Returns:
        x: inputted 'x' value
        y: f(x) evaluated 
        dotted_y: dotted version of f(x)
    '''
    # required x value
    x = f[0]
    
    y = 0
    dotted_y = 0
    
    for o in f[1:]:
        print(o)
    
    return x, y, dotted_y