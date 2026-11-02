"""Fixed workload protocol shared by every optimization phase."""
import re
import subprocess
import sys
import tempfile
from decimal import Decimal, InvalidOperation

SYSTEM = ('Solve the math problem. You must first call the python tool to calculate '
          'the answer with a Python program that prints the result. After seeing the '
          'tool output, give the final answer in \\boxed{N}.')
CONCISE_SYSTEM = ('Solve the math problem using the python tool. First respond with only '
                  'a python tool call containing a short program that prints the answer. '
                  'Do not explain before the tool call. After seeing the tool output, '
                  'reply with only the numeric answer in \\boxed{N}.')
MINIMAL_SYSTEM = ('Solve the math problem using the python tool. First respond with only '
                  'a python tool call containing the shortest executable Python program '
                  'that prints the answer. Use no comments or explanations. After seeing '
                  'the tool output, reply with only the numeric answer in \\boxed{N}.')
TOOLS = [{'type': 'function', 'function': {'name': 'python',
    'description': 'Execute Python code and return printed output.',
    'parameters': {'type': 'object', 'properties': {'code': {'type': 'string'}},
                   'required': ['code']}}}]

def parse_call(text):
    calls = re.findall(r'<tool_call>\s*<function=python>(.*?)</function>\s*</tool_call>', text, re.S)
    if not calls:
        return None
    code = re.search(r'<parameter=code>\s*(.*?)\s*</parameter>', calls[0], re.S)
    return code.group(1) if code else None

def execute(code):
    # Separate disposable working directory; time and output bounded.
    with tempfile.TemporaryDirectory(prefix='gsm8k-tool-') as cwd:
        try:
            p = subprocess.run([sys.executable, '-I', '-c', code], cwd=cwd,
                               capture_output=True, text=True, timeout=10)
            return (p.stdout + p.stderr)[-8192:]
        except subprocess.TimeoutExpired:
            return 'Python execution timed out after 10 seconds.'

def answer(text):
    boxes = re.findall(r'\\boxed\{([^{}]+)\}', text)
    if not boxes:
        return None
    return boxes[-1].strip().replace(',', '')

def correct(pred, expected):
    if pred is None:
        return False
    try:
        return Decimal(pred) == Decimal(str(expected).replace(',', ''))
    except InvalidOperation:
        return pred == str(expected)
