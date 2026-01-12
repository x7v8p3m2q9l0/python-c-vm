# Quick Reference Card
## Python-to-C Transpiler
# NOT UP-TO-DATE
---

## Installation

```bash
# Requirements: Python 3.7+, GCC
# No pip packages needed - uses stdlib only!

# Verify compiler
gcc --version
```

---

## Basic Usage

```bash
# Run demo (no arguments)
python main.py

# Test VM
python main.py --test-vm

# Compile with defaults (STANDARD security)
python main.py mycode.py --save-python output.py

# Use the generated module
python output.py 0 10    # Call function 0 with arg 10
```

---

## Security Levels

```bash
--security MINIMAL     # Fastest (5% overhead, 0.5s compile)
--security STANDARD    # Recommended (15% overhead, 1s) ← DEFAULT
--security AGGRESSIVE  # Strong (30% overhead, 1.5s)
--security PARANOID    # Maximum (50% overhead, 2.5s)
```

---

## Common Commands

```bash
# Show generated C code
python main.py code.py --show-c

# Show function mapping
python main.py code.py --show-metadata

# Test a function directly
python main.py code.py --call 0 --args 10 20

# Maximum security compilation
python main.py code.py --security PARANOID \
    --save-python secure.py

# Hardware binding
python main.py code.py --hardware-binding \
    --save-python bound.py

# Save encrypted container
python main.py code.py --save-bytes output.bin

# Keep temp files for debugging
python main.py code.py --keep-temp
```

---

## NEW Commands (v6.0.0)

```bash
# Generate standalone module (NEW!)
python main.py code.py --p2c-s2c standalone.py

# Test enhanced VM (NEW!)
python main.py --test-vm

# Combine features
python main.py code.py --p2c-s2c out.py --security AGGRESSIVE

# Disable optimizations
python main.py code.py --no-optimize --save-python debug.py
```

---

## Function Calling

### Original Method (Index-Based)
Generated with `--save-python`:

```python
import mymodule

# Call function by index (0, 1, 2, ...)
result = mymodule.call(0, 10)     # First function
result = mymodule.call(1, 20, 30) # Second function
```

From command line:
```bash
python mymodule.py 0 10       # function 0, arg 10
python mymodule.py 1 20 30    # function 1, args 20, 30
```

### NEW Method (p2c_s2c)
Generated with `--p2c-s2c`:

```python
from mymodule import factorial, fibonacci

# Call by name!
result = factorial(10)
result = fibonacci(20)
```

From command line:
```bash
python mymodule.py 0 10       # Same index-based
# OR use the convenience functions in code
```

---

## Supported Python Features

**Supported**:
- Integer arithmetic (int64)
- While loops
- If/else statements  
- Function calls (including recursive)
- Comparisons (<, >, <=, >=, ==, !=)
- Arithmetic operators (+, -, *, /, %)
- Bitwise operators (&, |, ^, <<, >>)
- Unary operators (-, not)
- Ternary expressions (a if cond else b)
- Augmented assignment (+=, -=, *=, /=, %=)

❌ **Not Supported**:
- Strings, lists, dicts, sets, tuples
- Classes, decorators, generators
- Imports (except in generated loader)
- Floats (int64 only)
- Exceptions, try/except/finally
- For loops (use while instead)
- Break, continue statements
- Lambda functions
- List comprehensions

---

## Example Code

```python
# example.py
def factorial(n):
    result = 1
    i = 2
    while i <= n:
        result = result * i
        i = i + 1
    return result

def fibonacci(n):
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

def gcd(a, b):
    while b != 0:
        temp = b
        b = a % b
        a = temp
    return a
```

Compile:
```bash
# Original method
python main.py example.py --save-python math.py

# NEW p2c_s2c method (easier distribution!)
python main.py example.py --p2c-s2c math.py
```

Use:
```bash
python math.py 0 10    # factorial(10) = 3628800
python math.py 1 20    # fibonacci(20) = 6765
python math.py 2 48 18 # gcd(48, 18) = 6
```

---

## Security Features Matrix

| Feature | MINIMAL | STANDARD | AGGRESSIVE | PARANOID |
|---------|---------|----------|------------|----------|
| Symbol stripping | ✅ | ✅ | ✅ | ✅ |
| Control-flow flattening | ❌ | ✅ | ✅ | ✅ |
| Opaque predicates | ❌ | ✅ | ✅ | ✅ |
| Data obfuscation | ❌ | ✅ | ✅ | ✅ |
| Anti-debug (C) | ❌ | ✅ | ✅ | ✅ |
| SecurityMonitor | ❌ | ❌ | ✅ | ✅ |
| Anti-VM detection | ❌ | ❌ | ✅ | ✅ |
| Anti-analysis tools | ❌ | ❌ | ✅ | ✅ |
| Integrity checks | ❌ | ❌ | ✅ | ✅ |
| Frequent checks | ❌ | ❌ | ❌ | ✅ |
| Timing detection | ❌ | ❌ | ❌ | ✅ |
| LTO optimization | ❌ | ❌ | ❌ | ✅ |

---

## VM Test Results

```bash
$ python main.py --test-vm

======================================================================
VM Test Suite
======================================================================

[Test] Arithmetic: 5 * 10 + 3
  ✓ Result: 53 (expected 53)

[Test] Comparison: 10 > 5
  ✓ Result: 1 (expected 1)

[Test] Bitwise: 15 & 7
  ✓ Result: 7 (expected 7)

[Test] Conditional jump
  ✓ Result: 100 (expected 100)

======================================================================
VM tests complete!
======================================================================
```

---

## Troubleshooting

### Compilation fails
```bash
# Check compiler
gcc --version

# Show generated C code
python main.py code.py --show-c

# Keep temp files
python main.py code.py --keep-temp

# Use minimal security for debugging
python main.py code.py --security MINIMAL --save-python out.py
```

### Module exits immediately (AGGRESSIVE/PARANOID)
```bash
# SecurityMonitor detected debugger/VM/analysis tool
# Solutions:
# 1. Use STANDARD or MINIMAL for development
python main.py code.py --security STANDARD --save-python out.py

# 2. Run without debugger
python mymodule.py 0 10  # Direct execution

# 3. Test on target hardware (if using --hardware-binding)
```

### Module too large
```bash
# Use minimal security for smaller files
python main.py code.py --security MINIMAL \
    --p2c-s2c output.py

# Typical sizes:
# MINIMAL: ~7 KB
# STANDARD: ~8 KB
# AGGRESSIVE: ~9 KB
# PARANOID: ~10 KB
```

### Wrong results
```bash
# Check variable redeclaration (fixed in v6.0.0)
# Verify logic with simple test
python main.py code.py --call 0 --args 10

# Compare with Python version
python code.py  # Original
python compiled.py 0 10  # Compiled
```

---

## Performance Guidelines

### Compilation Time
- MINIMAL: 0.5s
- STANDARD: 1s
- AGGRESSIVE: 1.5s
- PARANOID: 2.5s

### Runtime Overhead
- MINIMAL: +5%
- STANDARD: +15%
- AGGRESSIVE: +30% (includes runtime checks)
- PARANOID: +50% (frequent checks)

### Binary Size
- Compiled .so: 14-15 KB
- p2c_s2c module: 7-10 KB (depending on security)
- Original .py: ~1 KB

### Memory Usage
- Stack: 1024 elements (VM)
- Registers: 32 (VM)
- Heap: Minimal (no dynamic allocation)

---

## File Outputs

### --save-python (Original Method)
```
Input:  mycode.py (1 KB)
Output: mylib.py  (150 KB)
   ├── Embedded encrypted container (~50 KB)
   │   ├── Compressed DLL
   │   └── Metadata (JSON)
   └── Obfuscated loader (~100 KB)
       ├── SecurityMonitor
       ├── MemoryLoader
       └── BinaryContainer unpacker
```

### --p2c-s2c (NEW Method)
```
Input:  mycode.py (1 KB)
Output: mylib.py  (7-10 KB)
   ├── Embedded compressed binary (base64)
   ├── Integrity check (SHA-256)
   ├── Simple loader
   └── Function wrappers
```

---

## Best Practices

### Development
1. **Start with MINIMAL** - Fast iterations
2. **Test unobfuscated first** - Verify logic works
3. **Use --show-c** - Inspect generated code
4. **Keep temp files** - Debug compilation issues

### Testing
1. **Test with --call** - Quick function validation
2. **Run demo mode** - Comprehensive tests
3. **Compare outputs** - Original vs compiled
4. **Check all security levels** - Ensure compatibility

### Production
1. **Use STANDARD** - Good balance for most cases
2. **AGGRESSIVE for sensitive code** - Extra protection
3. **PARANOID for critical algorithms** - Maximum security
4. **Hardware binding** - Lock to specific machine
5. **Use p2c_s2c** - Easier distribution (single file)

### Distribution
1. **Use p2c_s2c** - Simpler for end users
2. **Include usage instructions** - `python module.py 0 args`
3. **Test on target platform** - Cross-platform compatibility
4. **Don't include source** - Only distribute compiled

---

## Comparing Methods

### --save-python (Original)
**Pros:**
- All security features
- Binary container format
- Hardware binding support
- SecurityMonitor included
- MemoryLoader (Linux memfd)

**Cons:**
- Larger file size (150 KB)
- More complex loader

**Use when:**
- Need maximum security
- Want hardware binding
- Need all advanced features

### --p2c-s2c (NEW)
**Pros:**
- Much smaller (7-10 KB)
- Simpler loader
- Easier to distribute
- Integrity checking
- Single file

**Cons:**
- No hardware binding
- No SecurityMonitor
- Simpler security

**Use when:**
- Distributing to users
- File size matters
- Simplicity preferred
- Don't need advanced security

---

## Getting Help

```bash
# Show all options
python main.py --help

# Test VM
python main.py --test-vm

# Run demo
python main.py

# Read docs
cat README.md
cat FEATURES.md
cat CHANGELOG.md
```

---

## What's New in v6.0.0

### Added
- **--p2c-s2c** flag (standalone modules)
- **--test-vm** flag (VM testing)
- **p2c_s2c()** function (API)

### Improved
- **Better error messages**
- **Comprehensive testing**
- **Updated documentation**

---

## Command Comparison

### v5.0.0 (Original)
```bash
python main.py code.py --save-python out.py
python main.py code.py --save-bytes out.bin
python main.py code.py --call 0 --args 10
```

### v6.0.0 - All Above PLUS:
```bash
python main.py code.py --p2c-s2c out.py
python main.py --test-vm
python main.py code.py --no-optimize
```

---

## Quick Decision Tree

```
Do you need maximum security?
├─ Yes
│  └─ Use: --save-python --security PARANOID --hardware-binding
│
├─ Need simple distribution?
│  └─ Use: --p2c-s2c --security STANDARD
│
├─ Development/testing?
│  └─ Use: --save-python --security MINIMAL --keep-temp
│
└─ Not sure?
   └─ Use: --p2c-s2c --security STANDARD (recommended)
```

---

## Phases 
**Phases Complete**:
-  Phase 1: Core Obfuscation
-  Phase 2: Binary Hardening
-  Phase 3: Binary Packaging
-  Phase 4: Memory Loading
-  Phase 5: Runtime Security
-  Phase 6: API Security
-  Phase 7: Python Obfuscation
-  Phase 8: Optimizations
-  Phase 10: VM & Hardware
-  p2c_s2c

---

## Version Info

- **Version**: 6.0.0
- **Python**: 3.7+
- **Platforms**: Linux, macOS, Windows
- **Dependencies**: None (stdlib only!)
- **Status**: Unsure

---

## Quick Examples

### Factorial
```bash
echo 'def factorial(n):
    result = 1
    i = 2
    while i <= n:
        result = result * i
        i = i + 1
    return result' > fact.py

python main.py fact.py --p2c-s2c fact_compiled.py
python fact_compiled.py 0 10
# Output: Result: 3628800
```

### GCD
```bash
echo 'def gcd(a, b):
    while b != 0:
        temp = b
        b = a % b
        a = temp
    return a' > gcd.py

python main.py gcd.py --p2c-s2c gcd_compiled.py
python gcd_compiled.py 0 48 18
# Output: Result: 6
```

---

*Quick Reference v6.0.0 - 2025-12-18*