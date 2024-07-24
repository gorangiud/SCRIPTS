from typing import *

def isValid(self, s: str) -> bool:
    open_par = ["(","[","{"]
    close_par = [")","]","}"]
    stack = []
    parenthesis = {
        "(" : ")",
        "[" : "]",
        "{" : "}"
    }
    if len(s) % 2 != 0:
        return False
    for i in range(len(s)):
        if s[i] in open_par:
            stack.insert(0,s[i])
        if s[i] in close_par:
            stack.append(s[i])
            if len(stack) > 1:
                if stack[-1] == parenthesis[stack[0]]:
                    stack.pop()
                    stack.pop(0)
                else:
                    return False
            else:
                return False
    if len(stack) > 0:
        return False
    return True