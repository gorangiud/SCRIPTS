from typing import *

def longestCommonPrefix(self, strs: List[str]) -> str:
    short = min(strs)
    string = ""
    if len(short) == 0:
        return string
    count = 0
    while(all([string+short[count] in i[:count+1] for i in strs])):
        string += short[count]
        if short == string:
            break
        count += 1
    return string
        