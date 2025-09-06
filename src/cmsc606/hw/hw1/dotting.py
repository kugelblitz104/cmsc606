import numpy as np

def dotting(f: list, other_info: float | None = None) -> tuple:
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
    x = other_info if other_info is not None else float(f[0][1])
    vars = [x]
    
    # evaluate y
    for ftuple in f[1:]:
        op = ftuple[0]
        args = ftuple[1:]
                
        u = args[0]
        v = args[1] if len(args) > 1 else None
        
        match op:
            case '+': vars.append(vars[u] + vars[v])
            case '-': vars.append(vars[u] - vars[v])
            case '*': vars.append(vars[u] * vars[v])
            case '/': vars.append(vars[u] / vars[v])
            case 'S': vars.append(np.square(vars[u]))
            case 'E': vars.append(np.exp(vars[u]))
            case 'L': vars.append(np.log(vars[u]))
            case 'X': vars.append(float(u))
            case 'C': vars.append(float(u))
            case _: raise ValueError(f"Invalid operation: {op}")

    y = float(vars[-1])
    
    # evaluate dotted y
    dotted = []
    for ftuple in f:
        op = ftuple[0]
        args = ftuple[1:]
        
        u = args[0]
        v = args[1] if len(args) > 1 else None
        
        match op:
            case '+': dotted.append(dotted[u] + dotted[v])
            case '-': dotted.append(dotted[u] - dotted[v])
            case '*': dotted.append(
                (dotted[u] * vars[v]) 
                + (vars[u] * dotted[v])
            )
            case '/': dotted.append(
                (
                    (dotted[u] * vars[v])
                    - (vars[u] * dotted[v])
                )
                / (np.square(vars[v]))
            )
            case 'S': dotted.append(
                (2.0 * vars[u])
                * (dotted[u])
            )
            case 'E': dotted.append(
                (np.exp(vars[u]))
                * (dotted[u])
            )
            case 'L': dotted.append(dotted[u] / vars[u])
            case 'X': dotted.append(1.0)
            case 'C': dotted.append(0.0)
            case _: raise ValueError(f"Invalid operation: {op}")


    dotted_y = float(dotted[-1])
    return x, y, dotted_y

def repeated_dotting(f: list, step_count: int = 10) -> tuple:
    x = float(f[0][1])
    y = 0
    dotted_y = 0
    for i in range(step_count):
        x, y, dotted_y = dotting(f, x)
        x -= dotted_y * 0.001
    return y, dotted_y