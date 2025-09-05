import cmsc606.hw.hw1._config as c

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
        return_tuple:
        - x: inputted 'x' value
        - y: f(x) evaluated 
        - dotted_y: dotted version of f(x)
    '''
    # required x value
    x = float(f[0][1])
    operations = c.functions.keys()
    vars = [x]
    
    for ftuple in f[1:]:
        op = ftuple[0]
        args = ftuple[1:]
        if op not in operations:
            raise ValueError(f"Invalid operation: {op}")
        
        if op in ('X', 'C'):
            vars.append(float(args[0]))
        else:
            func = c.functions[op]
            vars.append(func(*[vars[x] for x in args]))

    y = float(vars[-1])
    
    
    dotted_y = 0
    return x, y, dotted_y