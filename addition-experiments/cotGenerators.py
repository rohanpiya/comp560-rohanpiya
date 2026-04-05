import random

def generate2DigitCoTExample():
    
    a = random.randint(10,99)
    b = random.randint(10,99)

    a1 = a % 10
    a10 = a // 10

    b1 = b % 10
    b10 = b // 10

    # ones column
    ones_sum = a1 + b1
    carry1 = ones_sum // 10
    ones_digit = ones_sum % 10

    # tens column
    tens_sum = a10 + b10 + carry1
    carry2 = tens_sum // 10
    tens_digit = tens_sum % 10

    final_answer = a + b

    example = (
        f"{a}+{b}="
        f"{a1}+{b1}={ones_sum};"
        f"{a10}+{b10}+{carry1}={tens_sum};"
        f"{final_answer}\n"
    )

    return example

def generate2DigitSimpleCoTExample():

    while True:

        a = random.randint(10,99)
        b = random.randint(10,99)

        if (a%10 + b%10) < 10:

            a1 = a%10
            a10 = a//10
            b1 = b%10
            b10 = b//10

            ones_sum = a1 + b1
            carry = 0
            tens_sum = a10 + b10

            final_answer = a + b

            return (
                f"{a}+{b}="
                f"{a1}+{b1}={ones_sum};"
                f"{a10}+{b10}+0={tens_sum};"
                f"{final_answer}\n"
            )

def generate2DigitCarryCoTExample():

    while True:

        a = random.randint(10,99)
        b = random.randint(10,99)

        if (a%10 + b%10) >= 10:

            a1 = a%10
            a10 = a//10
            b1 = b%10
            b10 = b//10

            ones_sum = a1 + b1
            carry = ones_sum // 10

            tens_sum = a10 + b10 + carry

            final_answer = a + b

            return (
                f"{a}+{b}="
                f"{a1}+{b1}={ones_sum};"
                f"{a10}+{b10}+{carry}={tens_sum};"
                f"{final_answer}\n"
            )
        
import random

def generate3DigitCoTExample():
    a = random.randint(100, 999)
    b = random.randint(100, 999)

    # digits
    a1, a10, a100 = a % 10, (a // 10) % 10, a // 100
    b1, b10, b100 = b % 10, (b // 10) % 10, b // 100

    # ones
    ones_sum = a1 + b1
    carry1 = ones_sum // 10
    ones_digit = ones_sum % 10

    # tens
    tens_sum = a10 + b10 + carry1
    carry2 = tens_sum // 10
    tens_digit = tens_sum % 10

    # hundreds
    hundreds_sum = a100 + b100 + carry2
    carry3 = hundreds_sum // 10
    hundreds_digit = hundreds_sum % 10

    final_answer = a + b

    example = (
        f"{a}+{b}="
        f"{a1}+{b1}={ones_sum};"
        f"{a10}+{b10}+{carry1}={tens_sum};"
        f"{a100}+{b100}+{carry2}={hundreds_sum};"
        f"{final_answer}\n"
    )

    return example

def generate3DigitNoCarry():
    while True:
        a = random.randint(100, 999)
        b = random.randint(100, 999)

        if (a % 10 + b % 10 < 10 and
            (a // 10 % 10 + b // 10 % 10) < 10 and
            (a // 100 + b // 100) < 10):

            return generate3DigitCoTExample()
