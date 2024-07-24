from typing import *

def isPalindrome(self, x: int) -> bool:
    rev=0
    temp = x
    while(x>0):
        dig=x%10
        rev=rev*10+dig
        x=x//10
    if(temp==rev):
        return True
    else:
        False