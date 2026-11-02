"""Balanced boxed parsing and conservative numeric/time normalization."""
import re
from decimal import Decimal, InvalidOperation

def answer(text):
    values=[]
    for match in re.finditer(r'\\boxed\s*\{',text):
        start=match.end()
        depth=1
        for i in range(start,len(text)):
            if text[i]=='{': depth+=1
            elif text[i]=='}': depth-=1
            if depth==0:
                values.append(text[start:i].strip())
                break
    return values[-1] if values else None

def normalize(value):
    value=str(value).strip().replace(',','')
    value=re.sub(r'\\(?:text|mathrm)\{([^{}]*)\}',r'\1',value).strip()
    # GSM8K's numeric label omits the meridiem and :00 for integral hours.
    time=re.fullmatch(r'(1[0-2]|[1-9]):00\s*(?:[aApP]\.?\s*[mM]\.?)?',value)
    if time:
        value=time.group(1)
    try:
        return Decimal(value)
    except InvalidOperation:
        return value

def correct(pred,expected):
    return pred is not None and normalize(pred)==normalize(expected)
