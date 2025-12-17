# P2C
<div align="center">

[🇻🇳 Tiếng Việt](#-phiên-bản-tiếng-việt) |
[🇨🇳 中文](#-中文版本简体) |
[🇺🇸 English](#p2c)

</div>

## 🎯 What Was Done

### 1. Fixed the VM ✅
### 2. Added p2c_s2c Function ✅
### 3. Enhanced CLI ✅
**NEW Commands:**
```bash
# Generate standalone module
python main.py input.py --p2c-s2c output.py

# Test enhanced VM
python main.py --test-vm
```

---

## 🚀 Quick Start

### Basic Usage

```bash
# Generate standalone Python module
python main.py input.py --save-python output.py

# With security level
python main.py input.py --save-python output.py --security PARANOID

# Hardware binding
python main.py input.py --save-python output.py --hardware-binding

# Call function directly
python main.py input.py --call 0 --args 10

# Save as binary container
python main.py input.py --save-bytes output.bin
```

### NEW Features

```bash
# Generate p2c_s2c standalone module
python main.py input.py --p2c-s2c output.py

# Test enhanced VM
python main.py --test-vm

# Combine features
python main.py input.py --p2c-s2c output.py --security AGGRESSIVE
```

---

## 🧪 Testing

### Test the VM

```bash
$ python main.py --test-vm

======================================================================
Enhanced VM Test Suite
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

### Test p2c_s2c

```bash
# Create test file
cat > test.py << 'EOF'
def factorial(n):
    result = 1
    i = 2
    while i <= n:
        result = result * i
        i = i + 1
    return result
EOF

# Generate standalone module
python main.py test.py --p2c-s2c factorial_compiled.py

# Use it!
python factorial_compiled.py 0 10
# Output: Result: 3628800
```

---

## 📋 What's New

### VM Things

**30+ Opcodes:**
- Stack operations: LOAD_CONST, LOAD_VAR, STORE_VAR
- Arithmetic: ADD, SUB, MUL, DIV, MOD, NEG
- Bitwise: AND, OR, XOR, NOT, SHL, SHR
- Comparisons: LT, LE, GT, GE, EQ, NE
- Control flow: JUMP, JUMP_IF_FALSE, JUMP_IF_TRUE
- Functions: CALL, RETURN
- Stack manipulation: DUP, POP, SWAP
- Special: HALT

**VM Methods:**
```python
vm = VirtualMachine()
vm.load_code(bytecode)         # Load program
result = vm.execute()          # Run with cycle limit
assembly = vm.disassemble()    # Debug view
```

### p2c_s2c Function

**High-Level API:**
```python
from main import p2c_s2c, compile_python_to_standalone_module

# Method 1: Get Python code as string
python_code = p2c_s2c(source, "mymodule", SecurityLevel.STANDARD)

# Method 2: Write directly to file
compile_python_to_standalone_module(
    source,
    "output.py",
    SecurityLevel.AGGRESSIVE,
    "mymodule"
)
```


---


## 🎓 How It Works

### p2c_s2c Workflow

```
Python Source Code
       ↓
┌──────────────────────┐
│  p2c_s2c Function    │
└──────────────────────┘
       ↓
┌──────────────────────┐
│  Transpile to C      │
│  (All original       │
│   obfuscation)       │
└──────────────────────┘
       ↓
┌──────────────────────┐
│  Compile with GCC    │
│  (Hardening flags)   │
└──────────────────────┘
       ↓
┌──────────────────────┐
│  Read Binary         │
└──────────────────────┘
       ↓
┌──────────────────────┐
│  Compress (zlib)     │
└──────────────────────┘
       ↓
┌──────────────────────┐
│  Encode (base64)     │
└──────────────────────┘
       ↓
┌──────────────────────┐
│  Generate Python     │
│  Module with:        │
│  - Embedded binary   │
│  - Integrity check   │
│  - ctypes loader     │
│  - Function wrappers │
└──────────────────────┘
       ↓
Standalone .py File
(Distributable!)
```

### VM Execution Flow

```
Bytecode Program
       ↓
┌──────────────────────┐
│  vm.load_code()      │
│  - Initialize PC     │
│  - Clear stack       │
│  - Reset variables   │
└──────────────────────┘
       ↓
┌──────────────────────┐
│  vm.execute()        │
│  - Fetch opcode      │
│  - Decode arguments  │
│  - Execute operation │
│  - Update state      │
│  - Repeat            │
└──────────────────────┘
       ↓
Return Result
```

---

## 📖 Examples

### Example 1: Complete Workflow

```bash
# 1. Create Python code
cat > algorithms.py << 'EOF'
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
EOF

# 2. Generate standalone module
python main.py algorithms.py --p2c-s2c algorithms_compiled.py --security STANDARD

# 3. Use it!
python algorithms_compiled.py 0 10    # factorial(10) = 3628800
python algorithms_compiled.py 1 20    # fibonacci(20) = 6765
python algorithms_compiled.py 2 48 18 # gcd(48,18) = 6
```

### Example 2: Maximum Security

```bash
# Create with PARANOID security + hardware binding
python main.py algorithms.py \
  --p2c-s2c secure_algorithms.py \
  --security PARANOID \
  --hardware-binding

# The generated module will:
# - Have maximum obfuscation
# - Be hardware-locked
# - Include anti-debug checks
# - Have control-flow flattening
# - Include junk code
```

### Example 3: Features 

```bash
python main.py code.py --save-python lib.py
python main.py code.py --save-bytes lib.bin
python main.py code.py --call 0 --args 42
python main.py code.py --show-metadata
```

---
---
```bash
# Before
python main.py input.py --save-python output.py

# After (exact same command works!)
python main.py input.py --save-python output.py

# Plus new features
python main.py input.py --p2c-s2c output.py
python main.py --test-vm
```

---

## 💡 Tips & Best Practices

### 1. Choose Right Security Level

```bash
# Development - Fast
--security MINIMAL

# Production - Balanced  
--security STANDARD

# Sensitive Code
--security AGGRESSIVE

# Critical Algorithms
--security PARANOID
```

### 2. Use p2c_s2c for Distribution

```bash
# For distributing to users
python main.py mycode.py --p2c-s2c mylib.py --security STANDARD

# Users can just run

# P.S: Function indices correspond to function definition order in the source file (0-based).
python mylib.py 0 args...

# Single file, no C compiler needed!
```

### 3. Test Before Deploying

```bash
# Always test the generated module
python generated.py 0 test_args

# Compare with original

# P.S: Function indices correspond to function definition order in the source file (0-based).

python original.py  # Expected output
python generated.py 0 args  # Should match
```

### 4. Use Original Commands for Advanced Features

```bash
# For binary containers
python main.py code.py --save-bytes lib.bin

# For hardware binding
python main.py code.py --save-python lib.py --hardware-binding

# For metadata inspection
python main.py code.py --show-metadata
```

---

## 🎯 What to Use When

### Use `--save-python` when:
- Need original's binary container format
- Want hardware binding
- Need to load from bytes later
- Want index-based API

### Use `--p2c-s2c` when:
- Want single standalone file
- Distributing to end users
- Don't need hardware binding
- Want simpler deployment

---
---
=======
---


# P2C

---

# 🇻🇳 Phiên Bản Tiếng Việt

## 🎯 Những Gì Đã Hoàn Thành

### 1. Sửa lỗi VM ✅

### 2. Thêm hàm p2c_s2c ✅

### 3. Nâng cấp CLI ✅

**Lệnh Mới:**

```bash
# Tạo module độc lập
python main.py input.py --p2c-s2c output.py

# Kiểm tra VM nâng cao
python main.py --test-vm
```

---

## 🚀 Bắt Đầu Nhanh

### Cách Dùng Cơ Bản

```bash
# Tạo module Python độc lập
python main.py input.py --save-python output.py

# Với mức bảo mật
python main.py input.py --save-python output.py --security PARANOID

# Ràng buộc phần cứng
python main.py input.py --save-python output.py --hardware-binding

# Gọi hàm trực tiếp
python main.py input.py --call 0 --args 10

# Lưu dạng container nhị phân
python main.py input.py --save-bytes output.bin
```

### Tính Năng Mới

```bash
# Tạo module độc lập bằng p2c_s2c
python main.py input.py --p2c-s2c output.py

# Kiểm tra VM
python main.py --test-vm

# Kết hợp tính năng
python main.py input.py --p2c-s2c output.py --security AGGRESSIVE
```

---

## 🧪 Kiểm Thử

### Kiểm Thử VM

```bash
$ python main.py --test-vm

======================================================================
Enhanced VM Test Suite
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

### Kiểm Thử p2c_s2c

```bash
cat > test.py << 'EOF'
def factorial(n):
    result = 1
    i = 2
    while i <= n:
        result = result * i
        i = i + 1
    return result
EOF

python main.py test.py --p2c-s2c factorial_compiled.py

python factorial_compiled.py 0 10
# Output: Result: 3628800
```

---

## 📋 Điểm Mới

### VM

**Hơn 30 Opcode:**

* Ngăn xếp: LOAD_CONST, LOAD_VAR, STORE_VAR
* Số học: ADD, SUB, MUL, DIV, MOD, NEG
* Bit: AND, OR, XOR, NOT, SHL, SHR
* So sánh: LT, LE, GT, GE, EQ, NE
* Điều khiển luồng: JUMP, JUMP_IF_FALSE, JUMP_IF_TRUE
* Hàm: CALL, RETURN
* Ngăn xếp nâng cao: DUP, POP, SWAP
* Đặc biệt: HALT

**API VM:**

```python
vm = VirtualMachine()
vm.load_code(bytecode)
result = vm.execute()
assembly = vm.disassemble()
```

### Hàm p2c_s2c

```python
from main import p2c_s2c, compile_python_to_standalone_module

python_code = p2c_s2c(source, "mymodule", SecurityLevel.STANDARD)

compile_python_to_standalone_module(
    source,
    "output.py",
    SecurityLevel.AGGRESSIVE,
    "mymodule"
)
```

---

## 🎓 Cách Hoạt Động

### Quy Trình p2c_s2c

Python → C → GCC → Binary → Nén → Base64 → Module Python độc lập

### Luồng Thực Thi VM

Bytecode → load_code → execute → Kết quả

---

## 💡 Mẹo & Thực Hành Tốt Nhất

### Chọn Mức Bảo Mật

```bash
MINIMAL   # Phát triển
STANDARD  # Sản xuất
AGGRESSIVE
PARANOID
```

### Khi Nào Dùng p2c_s2c

* Phát hành cho người dùng
* 1 file duy nhất
* Không cần trình biên dịch C

---

# 🇨🇳 中文版本（简体）

## 🎯 已完成内容

### 1. 修复虚拟机（VM）✅

### 2. 添加 p2c_s2c 函数 ✅

### 3. 增强命令行接口（CLI）✅

**新增命令：**

```bash
# 生成独立模块
python main.py input.py --p2c-s2c output.py

# 测试增强版 VM
python main.py --test-vm
```

---

## 🚀 快速开始

### 基本用法

```bash
# 生成独立 Python 模块
python main.py input.py --save-python output.py

# 指定安全级别
python main.py input.py --save-python output.py --security PARANOID

# 硬件绑定
python main.py input.py --save-python output.py --hardware-binding

# 直接调用函数
python main.py input.py --call 0 --args 10

# 保存为二进制容器
python main.py input.py --save-bytes output.bin
```

### 新功能

```bash
python main.py input.py --p2c-s2c output.py
python main.py --test-vm
python main.py input.py --p2c-s2c output.py --security AGGRESSIVE
```

---

## 🧪 测试

### VM 测试

```bash
$ python main.py --test-vm

[Test] 算术运算: 5 * 10 + 3 → 53
[Test] 比较运算: 10 > 5 → 1
[Test] 位运算: 15 & 7 → 7
[Test] 条件跳转 → 100
```

### p2c_s2c 测试

```bash
python main.py test.py --p2c-s2c factorial_compiled.py
python factorial_compiled.py 0 10
```

---

## 📋 新特性

### 虚拟机（VM）

**30+ 指令：**

* 栈操作、算术、位运算、比较
* 控制流、函数调用

**VM API：**

```python
vm.load_code(bytecode)
vm.execute()
vm.disassemble()
```

### p2c_s2c API

```python
p2c_s2c(source, "mymodule", SecurityLevel.STANDARD)
```

---

## 🎓 工作原理

Python 源码 → C 转译 → GCC 编译 → 二进制 → 压缩 → Base64 → 独立 Python 模块

---

## 💡 使用建议

### 安全级别

```text
MINIMAL    开发
STANDARD   生产
AGGRESSIVE 高安全
PARANOID   最高安全
```

### 使用 p2c_s2c 的场景

* 分发给终端用户
* 单文件部署
* 无需 C 编译环境

---
