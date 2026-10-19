"""Balanced boxed parsing with 12-hour and 24-hour integral-time normalization."""
import re
from decimal import Decimal, InvalidOperation

from scoring_v2 import answer


def normalize(value):
    value = str(value).strip().replace(',', '')
    value = re.sub(r'\\(?:text|mathrm)\{([^{}]*)\}', r'\1', value).strip()

    time12 = re.fullmatch(r'(1[0-2]|[1-9]):00\s*(?:[aApP]\.?\s*[mM]\.?)?', value)
    if time12:
        value = time12.group(1)
    else:
        time24 = re.fullmatch(r'(?:[01]?\d|2[0-3]):00', value)
        if time24:
            hour = int(value.split(':', 1)[0])
            value = str(hour % 12 or (12 if hour == 12 else 0))

    try:
        return Decimal(value)
    except InvalidOperation:
        return value


def correct(pred, expected):
    return pred is not None and normalize(pred) == normalize(expected)
