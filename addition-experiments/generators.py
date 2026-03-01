import random

def generateExample(min_val, max_val):
    a = random.randint(min_val, max_val)
    b = random.randint(min_val, max_val)
    return f"{a}+{b}={a+b}\n"

def generateNDigitExample(numDigits):
    upperLimit = 10 ** (numDigits+1) - 1
    lowerLimit = 10 ** numDigits
    return generateExample(lowerLimit, upperLimit)

def generate1DigitSimpleExamples():
    while True:
        a = random.randint(0, 9)
        b = random.randint(0, 9)

        (a1,a10) = a % 10, a // 10
        (b1,b10) = b % 10, b // 10

        if a+b < 10:
            return f"{a}+{b}={a+b}\n"
    
def generate1DigitCarryExamples():
    while True:
        a = random.randint(0, 9)
        b = random.randint(0, 9)

        if a+b >= 10:
            return f"{a}+{b}={a+b}\n"
        
def generate2DigitSimpleExamples():
    while True:
        a = random.randint(10, 99)
        b = random.randint(10, 99)

        a1, a10 = a % 10, a // 10
        b1, b10 = b % 10, b // 10

        carry_ones = (a1 + b1) >= 10
        tens_sum = a10 + b10 + (1 if carry_ones else 0)
        carry_tens = tens_sum >= 10

        if not carry_ones and not carry_tens:
            return f"{a}+{b}={a+b}\n"


def generate2DigitCarryExamples():
    while True:
        a = random.randint(10, 99)
        b = random.randint(10, 99)

        a1, a10 = a % 10, a // 10
        b1, b10 = b % 10, b // 10

        carry_ones = (a1 + b1) >= 10
        tens_sum = a10 + b10 + (1 if carry_ones else 0)
        carry_tens = tens_sum >= 10

        if carry_ones or carry_tens:
            return f"{a}+{b}={a+b}\n"

def generateComplexExamples():
        a = random.randint(1000, 9999)
        b = random.randint(10, 9999)
        return f"{a}+{b}={a+b}\n"