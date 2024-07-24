from typing import *

def sortPeople(self, names: List[str], heights: List[int]) -> List[str]:
    return reversed([name for heigth, name in sorted(zip(heights,names))])