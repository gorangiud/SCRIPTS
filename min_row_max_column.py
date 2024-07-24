from typing import *

import numpy as np

def luckyNumbers (self, matrix: List[List[int]]) -> List[int]:
    matrix = np.array(matrix)
    for i in range(len(matrix)):
        if max(matrix[:,int(np.where(matrix == min(matrix[i,:]))[1])]) == min(matrix[i,:]):
            List = [int(min(matrix[i,:]))]
            return List