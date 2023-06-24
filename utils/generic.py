import numpy as np

def vertices2nparray(vertex):
    return np.asarray([vertex['x'],vertex['y'],vertex['z']]).T