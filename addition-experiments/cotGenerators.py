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