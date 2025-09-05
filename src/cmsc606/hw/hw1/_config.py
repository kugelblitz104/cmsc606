import numpy as np

functions = {
    "+": lambda x, y: x + y,
    "-": lambda x, y: x - y,
    "*": lambda x, y: x * y,
    "/": lambda x, y: x / y,
    "S": lambda x: np.square(x),
    "E": lambda x: np.exp(x),
    "L": lambda x: np.log(x),
    "X": lambda x: x,
    "C": lambda x: x
}