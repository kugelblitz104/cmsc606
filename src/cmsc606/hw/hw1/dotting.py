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
        return_tuple:
        - x: inputted 'x' value
        - y: f(x) evaluated 
        - dotted_y: dotted version of f(x)
    '''
    # required x value
    x = float(f[0][1])
    vars = [x]
    
    # evaluate y
    for ftuple in f[1:]:
        op = ftuple[0]
        args = ftuple[1:]
                
        match op:
            case '+': vars.append(vars[args[0]] + vars[args[1]])
            case '-': vars.append(vars[args[0]] - vars[args[1]])
            case '*': vars.append(vars[args[0]] * vars[args[1]])
            case '/': vars.append(vars[args[0]] / vars[args[1]])
            case 'S': vars.append(np.square(vars[args[0]]))
            case 'E': vars.append(np.exp(vars[args[0]]))
            case 'L': vars.append(np.log(vars[args[0]]))
            case 'X': vars.append(float(args[0]))
            case 'C': vars.append(float(args[0]))
            case _: raise ValueError(f"Invalid operation: {op}")

    y = float(vars[-1])
    
    # evaluate dotted y
    dotted = []
    for ftuple in f:
        op = ftuple[0]
        args = ftuple[1:]
        
        match op:
            case '+': dotted.append(dotted[args[0]] + dotted[args[1]])
            case '-': dotted.append(dotted[args[0]] - dotted[args[1]])
            case '*': dotted.append(
                (dotted[args[0]] * vars[args[1]]) 
                + (vars[args[0]] * dotted[args[1]])
            )
            case '/': dotted.append(
                (
                    (dotted[args[0]] * vars[1])
                    - (vars[args[0] * dotted[1]])
                )
                / (np.square(args[1]))
            )
            case 'S': dotted.append(
                (2.0 * vars[args[0]])
                * (dotted[args[0]])
            )
            case 'E': dotted.append(
                (np.exp(args[0]))
                * (dotted[args[0]])
            )
            case 'L': dotted.append(dotted[args[0]] / vars[args[0]])
            case 'X': dotted.append(1.0)
            case 'C': dotted.append(0.0)
            case _: raise ValueError(f"Invalid operation: {op}")


    dotted_y = float(dotted[-1])
    return x, y, dotted_y