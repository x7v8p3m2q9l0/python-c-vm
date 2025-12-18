"""
Example Python code for main.py transpiler
Version 6.0.0 Complete Edition

Demonstrates supported language features and test cases
"""

def factorial(n):
    """
    Calculate factorial of n
    
    Test: factorial(10) = 3,628,800
    Test: factorial(20) = 2,432,902,008,176,640,000
    """
    result = 1
    i = 2
    while i <= n:
        result = result * i
        i = i + 1
    return result

def fibonacci(n):
    """
    Calculate nth Fibonacci number
    
    Test: fibonacci(10) = 55
    Test: fibonacci(20) = 6,765
    Test: fibonacci(30) = 832,040
    """
    if n <= 1:
        return n
    a = 0
    b = 1
    i = 2
    while i <= n:
        temp = a + b
        a = b
        b = temp
        i = i + 1
    return b

def power(base, exp):
    """
    Calculate base^exp
    
    Test: power(2, 10) = 1,024
    Test: power(5, 5) = 3,125
    Test: power(3, 7) = 2,187
    """
    result = 1
    i = 0
    while i < exp:
        result = result * base
        i = i + 1
    return result

def gcd(a, b):
    """
    Calculate greatest common divisor using Euclidean algorithm
    
    Test: gcd(48, 18) = 6
    Test: gcd(100, 35) = 5
    Test: gcd(17, 19) = 1 (coprime)
    """
    while b != 0:
        temp = b
        b = a % b
        a = temp
    return a

def sum_range(start, end):
    """
    Sum integers from start to end (inclusive)
    
    Test: sum_range(1, 10) = 55
    Test: sum_range(1, 100) = 5,050
    """
    total = 0
    i = start
    while i <= end:
        total = total + i
        i = i + 1
    return total

def is_prime(n):
    """
    Check if n is prime
    Returns 1 for prime, 0 for not prime
    
    Test: is_prime(2) = 1
    Test: is_prime(17) = 1
    Test: is_prime(100) = 0
    Test: is_prime(97) = 1
    """
    if n <= 1:
        return 0
    if n <= 3:
        return 1
    if n % 2 == 0:
        return 0
    i = 3
    while i * i <= n:
        if n % i == 0:
            return 0
        i = i + 2
    return 1

def collatz_steps(n):
    """
    Count steps in Collatz sequence until reaching 1
    
    Test: collatz_steps(1) = 0
    Test: collatz_steps(2) = 1
    Test: collatz_steps(16) = 4
    Test: collatz_steps(27) = 111
    """
    steps = 0
    while n != 1:
        if n % 2 == 0:
            n = n / 2
        else:
            n = 3 * n + 1
        steps = steps + 1
    return steps

def lcm(a, b):
    """
    Calculate least common multiple
    
    Test: lcm(12, 18) = 36
    Test: lcm(21, 6) = 42
    """
    # LCM(a,b) = (a*b) / GCD(a,b)
    # First calculate GCD
    x = a
    y = b
    while y != 0:
        temp = y
        y = x % y
        x = temp
    # x now contains GCD
    return (a * b) / x

def absolute(n):
    """
    Calculate absolute value
    
    Test: absolute(5) = 5
    Test: absolute(-5) = 5
    Test: absolute(0) = 0
    """
    if n < 0:
        return -n
    return n

def max_of_two(a, b):
    """
    Return maximum of two numbers
    
    Test: max_of_two(5, 10) = 10
    Test: max_of_two(100, 50) = 100
    Test: max_of_two(7, 7) = 7
    """
    if a > b:
        return a
    return b

def min_of_two(a, b):
    """
    Return minimum of two numbers
    
    Test: min_of_two(5, 10) = 5
    Test: min_of_two(100, 50) = 50
    Test: min_of_two(7, 7) = 7
    """
    if a < b:
        return a
    return b

def count_digits(n):
    """
    Count number of digits in n
    
    Test: count_digits(123) = 3
    Test: count_digits(1000) = 4
    Test: count_digits(5) = 1
    """
    if n == 0:
        return 1
    count = 0
    if n < 0:
        n = -n
    while n > 0:
        count = count + 1
        n = n / 10
    return count

def reverse_digits(n):
    """
    Reverse the digits of n
    
    Test: reverse_digits(123) = 321
    Test: reverse_digits(1000) = 1
    Test: reverse_digits(505) = 505
    """
    result = 0
    while n > 0:
        digit = n % 10
        result = result * 10 + digit
        n = n / 10
    return result

def digital_root(n):
    """
    Calculate digital root (repeated sum of digits until single digit)
    
    Test: digital_root(38) = 2  (3+8=11, 1+1=2)
    Test: digital_root(999) = 9
    Test: digital_root(123) = 6
    """
    while n >= 10:
        sum = 0
        while n > 0:
            sum = sum + (n % 10)
            n = n / 10
        n = sum
    return n

def triangle_number(n):
    """
    Calculate nth triangle number (1+2+3+...+n)
    
    Test: triangle_number(5) = 15
    Test: triangle_number(10) = 55
    Test: triangle_number(100) = 5,050
    """
    return (n * (n + 1)) / 2

# ============================================================================
# USAGE INSTRUCTIONS
# ============================================================================

"""
COMPILE THIS FILE:

Method 1: Original (--save-python)
    python main.py example.py --save-python example_compiled.py

Method 2: NEW p2c_s2c (--p2c-s2c)  
    python main.py example.py --p2c-s2c example_compiled.py

Method 3: With Security Levels
    python main.py example.py --p2c-s2c example.py --security STANDARD
    python main.py example.py --p2c-s2c example.py --security AGGRESSIVE
    python main.py example.py --p2c-s2c example.py --security PARANOID

USE COMPILED MODULE:

Command Line:
    python example_compiled.py 0 10         # factorial(10)
    python example_compiled.py 1 20         # fibonacci(20)
    python example_compiled.py 2 2 10       # power(2, 10)
    python example_compiled.py 3 48 18      # gcd(48, 18)
    python example_compiled.py 4 1 100      # sum_range(1, 100)
    python example_compiled.py 5 17         # is_prime(17)
    python example_compiled.py 6 27         # collatz_steps(27)

In Python:
    from example_compiled import factorial, fibonacci, gcd
    
    print(factorial(10))      # 3628800
    print(fibonacci(20))      # 6765
    print(gcd(48, 18))        # 6

FUNCTION INDEX:
    0 - factorial(n)
    1 - fibonacci(n)
    2 - power(base, exp)
    3 - gcd(a, b)
    4 - sum_range(start, end)
    5 - is_prime(n)
    6 - collatz_steps(n)
    7 - lcm(a, b)
    8 - absolute(n)
    9 - max_of_two(a, b)
    10 - min_of_two(a, b)
    11 - count_digits(n)
    12 - reverse_digits(n)
    13 - digital_root(n)
    14 - triangle_number(n)

TEST VM:
    python main.py --test-vm

SHOW GENERATED C CODE:
    python main.py example.py --show-c

SHOW FUNCTION METADATA:
    python main.py example.py --show-metadata

DIRECT TESTING:
    python main.py example.py --call 0 --args 10

KEEP TEMP FILES:
    python main.py example.py --keep-temp --p2c-s2c out.py
"""

# ============================================================================
# EXPECTED TEST RESULTS
# ============================================================================

"""
factorial(10) = 3,628,800
factorial(20) = 2,432,902,008,176,640,000
fibonacci(10) = 55
fibonacci(20) = 6,765
fibonacci(30) = 832,040
power(2, 10) = 1,024
power(5, 5) = 3,125
gcd(48, 18) = 6
gcd(100, 35) = 5
sum_range(1, 10) = 55
sum_range(1, 100) = 5,050
is_prime(17) = 1
is_prime(100) = 0
collatz_steps(27) = 111
lcm(12, 18) = 36
absolute(-5) = 5
max_of_two(5, 10) = 10
min_of_two(5, 10) = 5
count_digits(123) = 3
reverse_digits(123) = 321
digital_root(38) = 2
triangle_number(10) = 55
"""