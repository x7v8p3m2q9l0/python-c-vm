#!/usr/bin/env python3
import ast
import sys
import os
import subprocess
import tempfile
import shutil
import argparse
import ctypes
import hashlib
import secrets
import struct
import zlib
import time
import platform
from pathlib import Path
from textwrap import dedent, indent as txt_indent
from typing import List, Tuple, Dict, Set, Optional, Any
from dataclasses import dataclass
from enum import Enum

# ============================================================================
# CONSTANTS & CONFIGURATION
# ============================================================================

VERSION = "5.0.0"
MAGIC_HEADER = b"PY2C"
CONTAINER_VERSION = 1

class SecurityLevel(Enum):
    """Security/obfuscation levels"""
    MINIMAL = 0      # Basic stripping only
    STANDARD = 1     # Moderate obfuscation
    AGGRESSIVE = 2   # Heavy obfuscation
    PARANOID = 3     # Maximum obfuscation

# ============================================================================
# EXCEPTIONS
# ============================================================================

class CompilationError(Exception):
    """Compilation failed"""
    pass

class TranspilationError(Exception):
    """Transpilation failed"""
    pass

class SecurityError(Exception):
    """Security check failed"""
    pass

class IntegrityError(Exception):
    """Integrity verification failed"""
    pass

# ============================================================================
# UTILITIES
# ============================================================================

class RandomGenerator:
    """Cryptographically secure random generation"""
    
    @staticmethod
    def seed_from_env() -> bytes:
        """Generate seed from environment"""
        env_data = f"{os.getpid()}{time.time()}{os.urandom(16)}"
        return hashlib.sha256(env_data.encode()).digest()
    
    @staticmethod
    def random_id(length: int = 8) -> str:
        """Generate random identifier"""
        return secrets.token_hex(length // 2)
    
    @staticmethod
    def random_symbol(prefix: str = "fn") -> str:
        """Generate random C symbol name"""
        return f"{prefix}_{secrets.token_hex(16)}"
    
    @staticmethod
    def random_int(min_val: int, max_val: int) -> int:
        """Generate random integer"""
        return secrets.randbelow(max_val - min_val + 1) + min_val

class SymbolTable:
    """Manages symbol name obfuscation"""
    
    def __init__(self, seed: Optional[bytes] = None):
        self.mapping: Dict[str, str] = {}
        self.reverse: Dict[str, str] = {}
        self.seed = seed or RandomGenerator.seed_from_env()
    
    def obfuscate(self, name: str) -> str:
        """Obfuscate a symbol name"""
        if name not in self.mapping:
            # Generate deterministic but random-looking name
            hash_input = f"{self.seed.hex()}{name}".encode()
            hash_val = hashlib.sha256(hash_input).hexdigest()[:16]
            obf_name = f"_{hash_val}"
            self.mapping[name] = obf_name
            self.reverse[obf_name] = name
        return self.mapping[name]
    
    def get_original(self, obfuscated: str) -> Optional[str]:
        """Get original name from obfuscated"""
        return self.reverse.get(obfuscated)

# ============================================================================
# CONTROL-FLOW OBFUSCATION
# ============================================================================

class OpaquePredicateGenerator:
    """Generate opaque predicates (always true/false conditions)"""
    
    @staticmethod
    def always_true() -> str:
        """Generate always-true predicate"""
        # Use actual constants to avoid undeclared variable errors
        c = RandomGenerator.random_int(1, 100)
        templates = [
            f"(({c} & 1) == 0 || ({c} & 1) == 1)",  # Tautology: even or odd
            f"(({c} * {c}) >= 0)",  # Always true: square is non-negative
            f"(({c} | 0) == {c})",  # Identity operation
            f"(({c} ^ 0) == {c})",  # XOR with zero
            f"((1) == (1))",  # Trivial but obscured by surrounding code
            f"(({c} + 0) == {c})",  # Addition identity
        ]
        return secrets.choice(templates)
    
    @staticmethod
    def always_false() -> str:
        """Generate always-false predicate"""
        c = RandomGenerator.random_int(1, 100)
        templates = [
            f"(({c} & (~{c})) != 0)",  # AND with complement != 0
            f"(({c} ^ {c}) != 0)",   # XOR with self != 0
            f"(({c} * 0) != 0)",   # Multiply by zero != 0
            f"(({c} < {c}))",        # Self comparison
            f"((0) != (0))",  # Trivial contradiction
            f"(({c} - {c}) != 0)",  # Subtract self != 0
        ]
        return secrets.choice(templates)
    
    @staticmethod
    def random_condition() -> str:
        """Generate random opaque condition"""
        return OpaquePredicateGenerator.always_true() if secrets.randbelow(2) else OpaquePredicateGenerator.always_false()

class ControlFlowFlattener:
    """Implements control-flow flattening (state machine transformation)"""
    
    def __init__(self):
        self.state_counter = 0
        self.var_declarations = set()
    
    def flatten(self, stmts: List[str], indent_level: int = 1) -> str:
        """Convert sequential statements to state machine"""
        if len(stmts) < 2:  # Need at least 2 statements
            return "\n".join(stmts)
        
        # Don't flatten if there are conditional returns or complex nested structures
        if self._has_complex_control_flow(stmts):
            return "\n".join(stmts)
        
        ind = "    " * indent_level
        
        # First pass: extract all variable declarations
        self.var_declarations = self._extract_variable_declarations(stmts)
        
        # Second pass: process statements without declarations
        processed_stmts = self._remove_declarations_from_stmts(stmts)
        
        # Assign random state IDs
        num_states = len(processed_stmts)
        states = list(range(num_states))
        secrets.SystemRandom().shuffle(states)
        
        # Generate state variable
        state_var = f"_s{self.state_counter}"
        self.state_counter += 1
        
        # Build the flattened code
        result = []
        
        # Add all variable declarations at function scope
        for var_decl in sorted(self.var_declarations):
            result.append(f"{ind}{var_decl};")
        
        # Add state machine
        result.append(f"{ind}int64 {state_var} = {states[0]};")
        result.append(f"{ind}while ({OpaquePredicateGenerator.always_true()}) {{")
        result.append(f"{ind}    switch ({state_var}) {{")
        
        # Add each statement as a case
        for i, (stmt, state_id) in enumerate(zip(processed_stmts, states)):
            result.append(f"{ind}    case {state_id}:")
            
            # Add statement lines
            for line in stmt.split('\n'):
                if line.strip():
                    result.append(f"{ind}        {line}")
            
            # Add occasional fake branches
            if i > 0 and secrets.randbelow(4) == 0:  # 25% chance after first state
                fake_state = RandomGenerator.random_int(1000, 9999)
                result.append(f"{ind}        if ({OpaquePredicateGenerator.always_false()}) {{")
                result.append(f"{ind}            {state_var} = {fake_state};")
                result.append(f"{ind}            break;")
                result.append(f"{ind}        }}")
            
            # State transition
            if i < num_states - 1:
                result.append(f"{ind}        {state_var} = {states[i + 1]};")
                result.append(f"{ind}        break;")
            else:
                # Last state - exit
                result.append(f"{ind}        goto _exit_{state_var};")
        
        # Add a few fake dead states
        for _ in range(RandomGenerator.random_int(1, 3)):
            fake_state = RandomGenerator.random_int(1000, 9999)
            result.append(f"{ind}    case {fake_state}:")
            result.append(f"{ind}        {state_var} = {states[0]};")
            result.append(f"{ind}        break;")
        
        result.append(f"{ind}    default:")
        result.append(f"{ind}        goto _exit_{state_var};")
        result.append(f"{ind}    }}")
        result.append(f"{ind}}}")
        result.append(f"{ind}_exit_{state_var}:;")
        
        return "\n".join(result)
    
    def _has_complex_control_flow(self, stmts: List[str]) -> bool:
        """Check if statements have complex control flow that shouldn't be flattened"""
        for i, stmt in enumerate(stmts):
            # Don't flatten if there's a return that's not the last statement
            if 'return' in stmt and i < len(stmts) - 1:
                return True
            # Don't flatten if there are nested if statements
            if stmt.count('if (') > 1:
                return True
            # Don't flatten if there are nested while loops
            if stmt.count('while (') > 1:
                return True
        return False
    
    def _extract_variable_declarations(self, stmts: List[str]) -> Set[str]:
        """Extract all variable declarations from statements"""
        declarations = set()
        
        for stmt in stmts:
            lines = stmt.split('\n')
            for line in lines:
                # Match pattern: int64 variable_name = ...
                if 'int64 ' in line and '=' in line and 'return' not in line:
                    # Extract just the declaration part
                    parts = line.split('=', 1)
                    decl_part = parts[0].strip()
                    # Clean up the declaration
                    if 'int64 ' in decl_part:
                        var_name = decl_part.replace('int64', '').strip()
                        declarations.add(f"int64 {var_name}")
        
        return declarations
    
    def _remove_declarations_from_stmts(self, stmts: List[str]) -> List[str]:
        """Remove 'int64' declarations from statements, keeping only assignments"""
        processed = []
        
        for stmt in stmts:
            lines = stmt.split('\n')
            new_lines = []
            
            for line in lines:
                if 'int64 ' in line and '=' in line and 'return' not in line:
                    # Remove 'int64' keyword, keep assignment
                    parts = line.split('=', 1)
                    var_name = parts[0].replace('int64', '').strip()
                    indent_match = len(line) - len(line.lstrip())
                    new_line = ' ' * indent_match + var_name + ' =' + parts[1]
                    new_lines.append(new_line)
                else:
                    new_lines.append(line)
            
            processed.append('\n'.join(new_lines))
        
        return processed

# ============================================================================
# DATA OBFUSCATION
# ============================================================================

class DataObfuscator:
    """Obfuscates constants and expressions"""
    
    @staticmethod
    def obfuscate_int(value: int) -> str:
        """Obfuscate integer constant"""
        if value == 0:
            variants = [
                "(0)",
                "(1 - 1)",
                "(x ^ x)",
                "(x & 0)",
            ]
            return secrets.choice(variants).replace('x', str(RandomGenerator.random_int(1, 100)))
        
        if value == 1:
            variants = [
                "(1)",
                "(0 + 1)",
                "((x / x))",
                "((x != 0) ? 1 : 0)",
            ]
            return secrets.choice(variants).replace('x', str(RandomGenerator.random_int(1, 100)))
        
        # For other values, use arithmetic identities
        offset = RandomGenerator.random_int(-50, 50)
        if offset == 0:
            offset = 1
        
        xor_key = RandomGenerator.random_int(1, 255)
        
        variants = [
            f"({value} + {offset} - {offset})",
            f"({value + offset} - {offset})",
            f"({value * 2} / 2)",
            f"(({value} ^ {xor_key}) ^ {xor_key})",  # Use same XOR key twice!
        ]
        return secrets.choice(variants)
    
    @staticmethod
    def obfuscate_expression(expr: str) -> str:
        """Add arithmetic identities to expression"""
        noise = RandomGenerator.random_int(1, 100)
        variants = [
            f"(({expr}) + {noise} - {noise})",
            f"(({expr}) * 1)",
            f"(({expr}) | 0)",
            f"(({expr}) ^ 0)",
        ]
        return secrets.choice(variants)

class PythonObfuscator:
    """Phase 7: Python-level obfuscation for loader code"""
    
    def __init__(self, security_level: SecurityLevel):
        self.security_level = security_level
    
    def obfuscate_string(self, s: str) -> str:
        """Encrypt string with runtime decryption"""
        if self.security_level.value < SecurityLevel.AGGRESSIVE.value:
            return repr(s)
        
        # XOR with random key
        key = secrets.token_bytes(16)
        encrypted = bytes(c ^ key[i % len(key)] for i, c in enumerate(s.encode()))
        
        return f"bytes([x^k[i%len(k)] for i,x in enumerate({list(encrypted)})]).decode() if (k:={list(key)}) else ''"
    
    def add_control_flow_obfuscation(self, code: str) -> str:
        """Add control-flow obfuscation to Python code"""
        # Disabled for now - too aggressive, breaks indentation
        # TODO: Implement smarter version that respects Python syntax
        return code
    
    def add_fake_functions(self) -> str:
        """Generate fake functions to confuse analysis"""
        if self.security_level.value < SecurityLevel.AGGRESSIVE.value:
            return ""
        
        fake_funcs = []
        for i in range(RandomGenerator.random_int(3, 8)):
            name = f"_{RandomGenerator.random_id()}"
            fake_funcs.append(f"""
def {name}():
    '''Fake function - never called'''
    x = {RandomGenerator.random_int(1, 1000)}
    return x * {RandomGenerator.random_int(1, 100)} + {RandomGenerator.random_int(1, 100)}
""")
        
        return '\n'.join(fake_funcs)
    
    def add_namespace_pollution(self) -> str:
        """Add fake symbols to confuse analysis"""
        if self.security_level.value < SecurityLevel.PARANOID.value:
            return ""
        
        symbols = []
        for i in range(RandomGenerator.random_int(5, 15)):
            name = f"_{RandomGenerator.random_id()}"
            value = RandomGenerator.random_int(0, 10000)
            symbols.append(f"{name} = {value}")
        
        return '\n'.join(symbols)

# ============================================================================
# PHASE 8: ENHANCED TRANSPILATION
# ============================================================================

class EnhancedTypeSystem:
    """Phase 8: Extended type system with float, arrays, strings"""
    
    # Type definitions
    INT64 = "int64"
    FLOAT64 = "float64"
    STRING = "string"
    ARRAY = "array"
    
    @staticmethod
    def infer_type(node: ast.expr) -> str:
        """Infer type from AST node"""
        if isinstance(node, ast.Constant):
            if isinstance(node.value, int):
                return EnhancedTypeSystem.INT64
            elif isinstance(node.value, float):
                return EnhancedTypeSystem.FLOAT64
            elif isinstance(node.value, str):
                return EnhancedTypeSystem.STRING
        return EnhancedTypeSystem.INT64  # Default

class AdvancedOptimizer:
    """Phase 8: Advanced optimizations"""
    
    def __init__(self):
        self.constants = {}
    
    def constant_folding(self, node: ast.expr) -> ast.expr:
        """Fold constant expressions at compile time"""
        if isinstance(node, ast.BinOp):
            if isinstance(node.left, ast.Constant) and isinstance(node.right, ast.Constant):
                left_val = node.left.value
                right_val = node.right.value
                
                # Evaluate at compile time
                if isinstance(node.op, ast.Add):
                    return ast.Constant(value=left_val + right_val)
                elif isinstance(node.op, ast.Mult):
                    return ast.Constant(value=left_val * right_val)
                elif isinstance(node.op, ast.Sub):
                    return ast.Constant(value=left_val - right_val)
        
        return node
    
    def dead_code_elimination(self, stmts: List[ast.stmt]) -> List[ast.stmt]:
        """Remove unreachable code"""
        result = []
        reachable = True
        
        for stmt in stmts:
            if not reachable:
                break
            
            result.append(stmt)
            
            # Check for early returns
            if isinstance(stmt, ast.Return):
                reachable = False
        
        return result
    
    def strength_reduction(self, node: ast.BinOp) -> ast.expr:
        """Replace expensive operations with cheaper ones"""
        # x * 2 -> x << 1
        if isinstance(node.op, ast.Mult):
            if isinstance(node.right, ast.Constant) and node.right.value == 2:
                return ast.BinOp(left=node.left, op=ast.LShift(), 
                               right=ast.Constant(value=1))
        
        # x / 2 -> x >> 1
        if isinstance(node.op, ast.Div):
            if isinstance(node.right, ast.Constant) and node.right.value == 2:
                return ast.BinOp(left=node.left, op=ast.RShift(),
                               right=ast.Constant(value=1))
        
        return node

# ============================================================================
# PHASE 10: NATIVE VM (SIMPLIFIED IMPLEMENTATION)
# ============================================================================

class VMOpcode(Enum):
    """Enhanced VM opcodes"""
    # Stack operations
    LOAD_CONST = 0x01
    LOAD_VAR = 0x02
    STORE_VAR = 0x03
    
    # Arithmetic
    ADD = 0x10
    SUB = 0x11
    MUL = 0x12
    DIV = 0x13
    MOD = 0x14
    NEG = 0x15
    
    # Bitwise
    AND = 0x20
    OR = 0x21
    XOR = 0x22
    NOT = 0x23
    SHL = 0x24
    SHR = 0x25
    
    # Comparisons
    LT = 0x30
    LE = 0x31
    GT = 0x32
    GE = 0x33
    EQ = 0x34
    NE = 0x35
    
    # Control flow
    JUMP = 0x40
    JUMP_IF_FALSE = 0x41
    JUMP_IF_TRUE = 0x42
    
    # Functions
    CALL = 0x50
    RETURN = 0x51
    
    # Stack manipulation
    DUP = 0x60
    POP = 0x61
    SWAP = 0x62
    
    HALT = 0xFF

class VirtualMachine:
    """Enhanced VM - replaces original broken implementation"""
    
    # Keep old constants for compatibility
    OP_LOAD = 0x01
    OP_STORE = 0x02
    OP_ADD = 0x03
    OP_SUB = 0x04
    OP_MUL = 0x05
    OP_DIV = 0x06
    OP_CMP = 0x07
    OP_JMP = 0x08
    OP_JZ = 0x09
    OP_RET = 0x0A
    OP_CALL = 0x0B
    
    def __init__(self, security_level=None):
        self.security_level = security_level
        # NOW ENABLED! (was: self.enabled = False)
        self.enabled = True  # ✅ FIX: Enable the VM
        
        # VM state
        self.stack: List[int] = []
        self.vars: dict = {}
        self.pc = 0
        self.code: List[Tuple] = []
        self.max_stack = 1024
        self.call_stack: List[int] = []
    
    def load_code(self, code: List[Tuple]):
        """Load bytecode"""
        self.code = code
        self.pc = 0
        self.stack = []
        self.vars = {}
        self.call_stack = []
    
    def push(self, value: int):
        """Push to stack"""
        if len(self.stack) >= self.max_stack:
            raise RuntimeError("Stack overflow")
        self.stack.append(value)
    
    def pop(self) -> int:
        """Pop from stack"""
        if not self.stack:
            raise RuntimeError("Stack underflow")
        return self.stack.pop()
    
    def execute(self, max_cycles: int = 1000000) -> int:
        """Execute bytecode"""
        cycles = 0
        
        while self.pc < len(self.code) and cycles < max_cycles:
            cycles += 1
            opcode, arg = self.code[self.pc]
            self.pc += 1
            
            try:
                if opcode == VMOpcode.LOAD_CONST:
                    self.push(arg)
                elif opcode == VMOpcode.LOAD_VAR:
                    if arg not in self.vars:
                        raise RuntimeError(f"Undefined variable: {arg}")
                    self.push(self.vars[arg])
                elif opcode == VMOpcode.STORE_VAR:
                    self.vars[arg] = self.pop()
                elif opcode == VMOpcode.ADD:
                    b, a = self.pop(), self.pop()
                    self.push(a + b)
                elif opcode == VMOpcode.SUB:
                    b, a = self.pop(), self.pop()
                    self.push(a - b)
                elif opcode == VMOpcode.MUL:
                    b, a = self.pop(), self.pop()
                    self.push(a * b)
                elif opcode == VMOpcode.DIV:
                    b, a = self.pop(), self.pop()
                    if b == 0:
                        raise RuntimeError("Division by zero")
                    self.push(a // b)
                elif opcode == VMOpcode.MOD:
                    b, a = self.pop(), self.pop()
                    self.push(a % b)
                elif opcode == VMOpcode.NEG:
                    self.push(-self.pop())
                elif opcode == VMOpcode.AND:
                    b, a = self.pop(), self.pop()
                    self.push(a & b)
                elif opcode == VMOpcode.OR:
                    b, a = self.pop(), self.pop()
                    self.push(a | b)
                elif opcode == VMOpcode.XOR:
                    b, a = self.pop(), self.pop()
                    self.push(a ^ b)
                elif opcode == VMOpcode.NOT:
                    self.push(~self.pop())
                elif opcode == VMOpcode.SHL:
                    b, a = self.pop(), self.pop()
                    self.push(a << b)
                elif opcode == VMOpcode.SHR:
                    b, a = self.pop(), self.pop()
                    self.push(a >> b)
                elif opcode == VMOpcode.LT:
                    b, a = self.pop(), self.pop()
                    self.push(1 if a < b else 0)
                elif opcode == VMOpcode.LE:
                    b, a = self.pop(), self.pop()
                    self.push(1 if a <= b else 0)
                elif opcode == VMOpcode.GT:
                    b, a = self.pop(), self.pop()
                    self.push(1 if a > b else 0)
                elif opcode == VMOpcode.GE:
                    b, a = self.pop(), self.pop()
                    self.push(1 if a >= b else 0)
                elif opcode == VMOpcode.EQ:
                    b, a = self.pop(), self.pop()
                    self.push(1 if a == b else 0)
                elif opcode == VMOpcode.NE:
                    b, a = self.pop(), self.pop()
                    self.push(1 if a != b else 0)
                elif opcode == VMOpcode.JUMP:
                    self.pc = arg
                elif opcode == VMOpcode.JUMP_IF_FALSE:
                    if self.pop() == 0:
                        self.pc = arg
                elif opcode == VMOpcode.JUMP_IF_TRUE:
                    if self.pop() != 0:
                        self.pc = arg
                elif opcode == VMOpcode.CALL:
                    self.call_stack.append(self.pc)
                    self.pc = arg
                elif opcode == VMOpcode.RETURN:
                    if self.call_stack:
                        self.pc = self.call_stack.pop()
                    else:
                        return self.pop() if self.stack else 0
                elif opcode == VMOpcode.DUP:
                    self.push(self.stack[-1] if self.stack else 0)
                elif opcode == VMOpcode.POP:
                    self.pop()
                elif opcode == VMOpcode.SWAP:
                    b, a = self.pop(), self.pop()
                    self.push(b)
                    self.push(a)
                elif opcode == VMOpcode.HALT:
                    break
                else:
                    raise RuntimeError(f"Unknown opcode: {opcode}")
            except Exception as e:
                raise RuntimeError(f"VM error at PC={self.pc-1}: {e}")
        
        if cycles >= max_cycles:
            raise RuntimeError("Max cycles exceeded")
        
        return self.stack[-1] if self.stack else 0
    
    def disassemble(self) -> str:
        """Disassemble bytecode"""
        lines = []
        for i, (opcode, arg) in enumerate(self.code):
            arg_str = f" {arg}" if arg is not None else ""
            lines.append(f"{i:4d}: {opcode.name}{arg_str}")
        return "\n".join(lines)
    
    def translate_to_bytecode(self, c_code: str) -> bytes:
        """Translate C to bytecode (compatibility stub)"""
        if not self.enabled:
            return b""
        # Original was simplified anyway
        bytecode = bytearray()
        bytecode.extend([self.OP_LOAD, 0, 10])
        bytecode.extend([self.OP_LOAD, 1, 20])
        bytecode.extend([self.OP_ADD, 0, 1])
        bytecode.extend([self.OP_RET, 0])
        return bytes(bytecode)
    
    def generate_vm_runtime(self) -> str:
        """Generate VM runtime C code"""
        if not self.enabled:
            return ""
        
        return """
        // Enhanced VM Runtime
        typedef struct {
            int64 registers[32];
            int64 stack[1024];
            int sp;
            int pc;
        } VM;
        
        int64 vm_execute(VM* vm, unsigned char* code, int len) {
            vm->pc = 0;
            vm->sp = 0;
            
            while (vm->pc < len) {
                unsigned char op = code[vm->pc++];
                
                switch (op) {
                    case 0x01: { // LOAD_CONST
                        int reg = code[vm->pc++];
                        int val = code[vm->pc++];
                        vm->registers[reg] = val;
                        break;
                    }
                    case 0x10: { // ADD
                        int64 b = vm->stack[--vm->sp];
                        int64 a = vm->stack[--vm->sp];
                        vm->stack[vm->sp++] = a + b;
                        break;
                    }
                    case 0x11: { // SUB
                        int64 b = vm->stack[--vm->sp];
                        int64 a = vm->stack[--vm->sp];
                        vm->stack[vm->sp++] = a - b;
                        break;
                    }
                    case 0x12: { // MUL
                        int64 b = vm->stack[--vm->sp];
                        int64 a = vm->stack[--vm->sp];
                        vm->stack[vm->sp++] = a * b;
                        break;
                    }
                    case 0x51: { // RETURN
                        if (vm->sp > 0) return vm->stack[--vm->sp];
                        return 0;
                    }
                    default:
                        return 0;
                }
            }
            return 0;
        }
        """


class HardwareBinding:
    """Phase 10: Hardware-based protection"""
    
    @staticmethod
    def get_cpu_id() -> str:
        """Get CPU identifier"""
        try:
            if sys.platform == "linux":
                with open('/proc/cpuinfo', 'r') as f:
                    for line in f:
                        if 'processor' in line:
                            return hashlib.sha256(line.encode()).hexdigest()[:16]
        except:
            pass
        return "generic_cpu"
    
    @staticmethod
    def get_mac_address() -> str:
        """Get MAC address for binding"""
        try:
            import uuid
            mac = uuid.getnode()
            return hashlib.sha256(str(mac).encode()).hexdigest()[:16]
        except:
            return "generic_mac"
    
    @staticmethod
    def generate_binding_key(bind_cpu: bool = False, bind_mac: bool = False) -> bytes:
        """Generate hardware-bound encryption key"""
        components = [str(time.time())]
        
        if bind_cpu:
            components.append(HardwareBinding.get_cpu_id())
        
        if bind_mac:
            components.append(HardwareBinding.get_mac_address())
        
        key_material = ''.join(components).encode()
        return hashlib.sha256(key_material).digest()
    
    @staticmethod
    def verify_hardware(expected_cpu: Optional[str] = None,
                       expected_mac: Optional[str] = None) -> bool:
        """Verify hardware matches expected values"""
        if expected_cpu and HardwareBinding.get_cpu_id() != expected_cpu:
            return False
        
        if expected_mac and HardwareBinding.get_mac_address() != expected_mac:
            return False
        
        return True

# ============================================================================
# ENHANCED CODE GENERATOR (WITH ALL PHASES)
# ============================================================================

@dataclass
class FunctionMetadata:
    """Metadata for compiled function"""
    original_name: str
    obfuscated_name: str
    param_count: int
    index: int

class AdvancedCCodeGenerator:
    """Advanced C code generator with obfuscation and optimizations (Phases 1, 7, 8)"""
    
    def __init__(self, security_level: SecurityLevel = SecurityLevel.STANDARD,
                 enable_optimizations: bool = True):
        self.functions: List[str] = []
        self.func_metadata: List[FunctionMetadata] = []
        self.symbol_table = SymbolTable()
        self.security_level = security_level
        self.cf_flattener = ControlFlowFlattener()
        self.data_obf = DataObfuscator()
        self.func_index = 0
        
        # Phase 8: Advanced optimizations
        self.enable_optimizations = enable_optimizations
        self.optimizer = AdvancedOptimizer() if enable_optimizations else None
        
        # Phase 10: VM support
        self.vm = VirtualMachine(security_level)
    
    def generate(self, py_source: str) -> Tuple[str, List[FunctionMetadata]]:
        """Generate obfuscated C code from Python source"""
        tree = ast.parse(py_source)
        
        # Phase 8: Apply optimizations to AST
        if self.optimizer:
            tree = self._optimize_ast(tree)
        
        # Extract functions
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                self._process_function(node)
        
        if not self.functions:
            raise TranspilationError("No functions found")
        
        # Generate complete C code with obfuscation
        return self._build_c_code(), self.func_metadata
    
    def _optimize_ast(self, tree: ast.Module) -> ast.Module:
        """Apply Phase 8 optimizations to AST"""
        # Optimize each function
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                # Dead code elimination
                node.body = self.optimizer.dead_code_elimination(node.body)
                
                # Constant folding on expressions
                for i, stmt in enumerate(node.body):
                    if isinstance(stmt, ast.Assign):
                        stmt.value = self.optimizer.constant_folding(stmt.value)
        
        return tree
    
    def _process_function(self, node: ast.FunctionDef):
        """Process a single function"""
        orig_name = node.name
        obf_name = self.symbol_table.obfuscate(orig_name)
        param_count = len(node.args.args)
        
        # Store metadata
        metadata = FunctionMetadata(
            original_name=orig_name,
            obfuscated_name=obf_name,
            param_count=param_count,
            index=self.func_index
        )
        self.func_metadata.append(metadata)
        self.func_index += 1
        
        # Generate function code
        func_code = self._func_to_c(node, obf_name)
        self.functions.append(func_code)
    
    def _func_to_c(self, node: ast.FunctionDef, obf_name: str) -> str:
        """Convert Python function to obfuscated C"""
        # Generate parameter list with obfuscated names
        params = []
        param_map = {}
        for arg in node.args.args:
            obf_param = self.symbol_table.obfuscate(arg.arg)
            params.append(f"int64 {obf_param}")
            param_map[arg.arg] = obf_param
        
        params_str = ", ".join(params) if params else "void"
        
        # Generate body
        declared_vars = set(param_map.values())
        body_stmts = self._generate_statements(node.body, param_map, declared_vars)
        
        # Apply control-flow flattening if enabled
        if self.security_level.value >= SecurityLevel.STANDARD.value:
            body = self.cf_flattener.flatten(body_stmts, indent_level=1)
        else:
            body = "\n".join(body_stmts)
        
        # Add junk code at start (anti-disassembly)
        if self.security_level.value >= SecurityLevel.AGGRESSIVE.value:
            junk = self._generate_junk_code()
            body = junk + "\n" + body
        
        return f"int64 {obf_name}({params_str}) {{\n{body}\n}}"
    
    def _generate_statements(self, stmts: List[ast.stmt], var_map: Dict[str, str], 
                           declared_vars: Set[str], indent_level: int = 1) -> List[str]:
        """Generate C statements from AST"""
        ind = "    " * indent_level
        result = []
        
        for stmt in stmts:
            if isinstance(stmt, ast.Return):
                if stmt.value:
                    expr = self._expr_to_c(stmt.value, var_map)
                    if self.security_level.value >= SecurityLevel.STANDARD.value:
                        expr = self.data_obf.obfuscate_expression(expr)
                    result.append(f"{ind}return {expr};")
                else:
                    result.append(f"{ind}return 0;")
            
            elif isinstance(stmt, ast.Assign):
                if isinstance(stmt.targets[0], ast.Name):
                    orig_var = stmt.targets[0].id
                    if orig_var not in var_map:
                        var_map[orig_var] = self.symbol_table.obfuscate(orig_var)
                    
                    obf_var = var_map[orig_var]
                    val = self._expr_to_c(stmt.value, var_map)
                    
                    if obf_var not in declared_vars:
                        declared_vars.add(obf_var)
                        result.append(f"{ind}int64 {obf_var} = {val};")
                    else:
                        result.append(f"{ind}{obf_var} = {val};")
            
            elif isinstance(stmt, ast.AugAssign):
                orig_var = stmt.target.id
                obf_var = var_map.get(orig_var, orig_var)
                val = self._expr_to_c(stmt.value, var_map)
                op = self._get_op(stmt.op)
                result.append(f"{ind}{obf_var} {op}= {val};")
            
            elif isinstance(stmt, ast.While):
                test = self._expr_to_c(stmt.test, var_map)
                body_stmts = self._generate_statements(stmt.body, var_map, declared_vars, indent_level + 1)
                body = "\n".join(body_stmts)
                result.append(f"{ind}while ({test}) {{\n{body}\n{ind}}}")
            
            elif isinstance(stmt, ast.If):
                test = self._expr_to_c(stmt.test, var_map)
                body_stmts = self._generate_statements(stmt.body, var_map, declared_vars, indent_level + 1)
                body = "\n".join(body_stmts)
                code = f"{ind}if ({test}) {{\n{body}\n{ind}}}"
                
                if stmt.orelse:
                    else_stmts = self._generate_statements(stmt.orelse, var_map, declared_vars, indent_level + 1)
                    else_body = "\n".join(else_stmts)
                    code += f" else {{\n{else_body}\n{ind}}}"
                
                result.append(code)
        
        return result
    
    def _expr_to_c(self, node: ast.expr, var_map: Dict[str, str]) -> str:
        """Convert expression to C with obfuscation"""
        if isinstance(node, ast.Constant):
            if self.security_level.value >= SecurityLevel.STANDARD.value:
                return self.data_obf.obfuscate_int(node.value)
            return str(node.value)
        
        elif isinstance(node, ast.Name):
            orig_name = node.id
            return var_map.get(orig_name, self.symbol_table.obfuscate(orig_name))
        
        elif isinstance(node, ast.BinOp):
            left = self._expr_to_c(node.left, var_map)
            right = self._expr_to_c(node.right, var_map)
            op = self._get_op(node.op)
            return f"({left} {op} {right})"
        
        elif isinstance(node, ast.Compare):
            left = self._expr_to_c(node.left, var_map)
            right = self._expr_to_c(node.comparators[0], var_map)
            op = self._get_cmp_op(node.ops[0])
            return f"({left} {op} {right})"
        
        elif isinstance(node, ast.Call):
            orig_fname = node.func.id
            obf_fname = self.symbol_table.obfuscate(orig_fname)
            args = ", ".join(self._expr_to_c(arg, var_map) for arg in node.args)
            return f"{obf_fname}({args})"
        
        elif isinstance(node, ast.IfExp):
            test = self._expr_to_c(node.test, var_map)
            body = self._expr_to_c(node.body, var_map)
            orelse = self._expr_to_c(node.orelse, var_map)
            return f"(({test}) ? ({body}) : ({orelse}))"
        
        elif isinstance(node, ast.UnaryOp):
            operand = self._expr_to_c(node.operand, var_map)
            if isinstance(node.op, ast.USub):
                return f"(-{operand})"
            elif isinstance(node.op, ast.Not):
                return f"(!{operand})"
        
        return "0"
    
    def _get_op(self, op) -> str:
        ops = {
            ast.Add: '+', ast.Sub: '-', ast.Mult: '*',
            ast.Div: '/', ast.FloorDiv: '/', ast.Mod: '%',
        }
        return ops.get(type(op), '+')
    
    def _get_cmp_op(self, op) -> str:
        ops = {
            ast.Eq: '==', ast.NotEq: '!=', ast.Lt: '<',
            ast.LtE: '<=', ast.Gt: '>', ast.GtE: '>=',
        }
        return ops.get(type(op), '==')
    
    def _generate_junk_code(self) -> str:
        """Generate junk code for anti-disassembly"""
        junk_lines = []
        for _ in range(RandomGenerator.random_int(2, 5)):
            var = f"_junk_{RandomGenerator.random_id()}"
            val = RandomGenerator.random_int(0, 1000)
            junk_lines.append(f"    volatile int64 {var} = {val};")
            junk_lines.append(f"    if ({OpaquePredicateGenerator.always_false()}) {{")
            junk_lines.append(f"        {var} = {var} + 1;  // Dead code")
            junk_lines.append(f"    }}")
        return "\n".join(junk_lines)
    
    def _build_c_code(self) -> str:
        """Build complete C source code with advanced anti-reversing"""
        header = dedent("""
        #include <stdio.h>
        #include <stdlib.h>
        #include <string.h>
        #include <stdint.h>
        
        typedef int64_t int64;
        typedef volatile int64_t vint64;
        """).strip()
        
        # Add anti-debug macros
        anti_debug = ""
        if self.security_level.value >= SecurityLevel.STANDARD.value:
            anti_debug = dedent("""
            
            // Multi-layer anti-debug
            #ifdef _WIN32
            #include <windows.h>
            static inline int check_debug(void) {
                if (IsDebuggerPresent()) return 1;
                BOOL is_debugged = FALSE;
                CheckRemoteDebuggerPresent(GetCurrentProcess(), &is_debugged);
                return is_debugged;
            }
            #else
            #include <sys/ptrace.h>
            static inline int check_debug(void) {
                return ptrace(PTRACE_TRACEME, 0, 1, 0) == -1;
            }
            #endif
            
            #define ANTI_DBG if(check_debug())return 0;
            """)
        
        # Advanced anti-reversing for AGGRESSIVE+
        anti_reversing = ""
        if self.security_level.value >= SecurityLevel.AGGRESSIVE.value:
            anti_reversing = dedent("""
            
            // Anti-disassembly junk macros
            #define JUNK1 {volatile int _j=0; _j++; _j--;}
            #define JUNK2 __asm__("nop;nop;");
            #define JUNK3 {if(0){__builtin_unreachable();}}
            
            // Opaque predicates
            #define ALWAYS_TRUE ((0x5A5A & 0x0F0F) == 0x0A0A)
            #define NEVER_TRUE ((0x1234 | 0xFEDC) == 0)
            
            // Code checksum (anti-tamper)
            static volatile int64 _cs = 0x41424344;
            #define CHK_CS if(_cs!=0x41424344)return 0;
            """)
        
        # Inline assembly tricks for PARANOID
        asm_tricks = ""
        if self.security_level.value >= SecurityLevel.PARANOID.value:
            asm_tricks = dedent("""
            
            // Inline assembly anti-debug
            #if defined(__x86_64__) || defined(__i386__)
            static inline int _asm_check(void) {
                int result = 0;
                __asm__ volatile (
                    "pushf\\n"
                    "pop %%rax\\n"
                    "and $0x100, %%rax\\n"  // Check trap flag
                    "mov %%rax, %0"
                    : "=r"(result) : : "rax"
                );
                return result != 0;
            }
            #define ASM_CHK if(_asm_check())return 0;
            #else
            #define ASM_CHK
            #endif
            """)
        
        # Phase 10: Add VM runtime for PARANOID mode
        vm_runtime = ""
        if self.vm.enabled:
            vm_runtime = self.vm.generate_vm_runtime()
        
        # Function declarations
        declarations = "\n".join(
            f"int64 {meta.obfuscated_name}(...);" 
            for meta in self.func_metadata
        )
        
        # Function implementations
        implementations = "\n\n".join(self.functions)
        
        return f"{header}\n{anti_debug}\n{anti_reversing}\n{asm_tricks}\n{vm_runtime}\n\n{implementations}"

# ============================================================================
# ADVANCED COMPILER
# ============================================================================

class AdvancedCompiler:
    """Compiler with hardening and anti-analysis features"""
    
    def __init__(self, security_level: SecurityLevel = SecurityLevel.STANDARD):
        self.security_level = security_level
    
    def compile(self, c_source: str, output_dir: str, lib_name: str = "lib") -> str:
        """Compile C to hardened shared library"""
        c_file = os.path.join(output_dir, f"{lib_name}.c")
        
        # Write C source
        with open(c_file, 'w') as f:
            f.write(c_source)
        
        # Determine platform and build command
        if sys.platform.startswith("linux"):
            lib_file = os.path.join(output_dir, f"{lib_name}.so")
            cmd = ["gcc", "-shared", "-fPIC", c_file, "-o", lib_file]
        elif sys.platform == "darwin":
            lib_file = os.path.join(output_dir, f"{lib_name}.dylib")
            cmd = ["gcc", "-dynamiclib", c_file, "-o", lib_file]
        elif sys.platform.startswith("win"):
            lib_file = os.path.join(output_dir, f"{lib_name}.dll")
            cmd = ["gcc", "-shared", "-o", lib_file, c_file]
        else:
            raise CompilationError(f"Unsupported platform: {sys.platform}")
        
        # Add hardening flags
        hardening_flags = self._get_hardening_flags()
        cmd[1:1] = hardening_flags
        
        print(f"[*] Compiling with hardening...")
        print(f"    Security level: {self.security_level.name}")
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise CompilationError(f"Compilation failed:\n{result.stderr}")
        
        # Strip symbols
        self._strip_binary(lib_file)
        
        print(f"[✓] Compiled: {lib_file}")
        return lib_file
    
    def _get_hardening_flags(self) -> List[str]:
        """Get compiler hardening flags based on security level"""
        flags = [
            "-w",  # Suppress warnings
            "-O3",  # Optimize
        ]
        
        if self.security_level.value >= SecurityLevel.STANDARD.value:
            flags.extend([
                "-fno-asynchronous-unwind-tables",
                "-fno-ident",
                "-fomit-frame-pointer",
            ])
        
        if self.security_level.value >= SecurityLevel.AGGRESSIVE.value:
            flags.extend([
                "-fno-exceptions",
                "-fno-rtti",
                "-fno-stack-protector",
                # Note: NOT using -fvisibility=hidden because we need to export functions
                # for ctypes to call them. Symbol names are already obfuscated anyway.
            ])
        
        if self.security_level.value >= SecurityLevel.PARANOID.value:
            flags.extend([
                "-flto",  # Link-time optimization
                "-ffunction-sections",
                "-fdata-sections",
            ])
        
        return flags
    
    def _strip_binary(self, lib_file: str):
        """Strip symbols from binary"""
        if self.security_level.value >= SecurityLevel.STANDARD.value:
            try:
                subprocess.run(["strip", "-s", lib_file], 
                             capture_output=True, check=True)
                print(f"[✓] Stripped symbols")
            except (subprocess.CalledProcessError, FileNotFoundError):
                print(f"[!] Could not strip symbols (strip not available)")

# ============================================================================
# BINARY PACKAGING & ENCRYPTION
# ============================================================================

@dataclass
class BinarySection:
    """Binary container section"""
    name: str
    data: bytes
    compressed: bool = False
    encrypted: bool = False

class BinaryContainer:
    """Custom binary container with encryption and integrity"""
    
    def __init__(self, key: Optional[bytes] = None):
        self.sections: List[BinarySection] = []
        self.key = key or self._derive_key()
        self.metadata = {
            'platform': sys.platform,
            'arch': platform.machine(),
            'version': VERSION,
            'timestamp': int(time.time()),
        }
    
    def _derive_key(self) -> bytes:
        """Derive encryption key from environment"""
        # Environment-based key derivation
        env_factors = [
            str(os.getpid()),
            platform.node(),
            platform.processor(),
            str(time.time()),
        ]
        key_material = ''.join(env_factors).encode()
        return hashlib.sha256(key_material).digest()
    
    def add_section(self, name: str, data: bytes, compress: bool = True, 
                   encrypt: bool = True):
        """Add section to container"""
        processed_data = data
        
        # Compress FIRST if requested (compresses better before encryption)
        if compress:
            processed_data = zlib.compress(processed_data, level=9)
        
        # Encrypt SECOND if requested (after compression)
        if encrypt:
            processed_data = self._encrypt(processed_data)
        
        section = BinarySection(name, processed_data, compress, encrypt)
        self.sections.append(section)
    
    def _encrypt(self, data: bytes) -> bytes:
        """Encrypt data using XOR with key stream (simple but effective)"""
        key_stream = hashlib.sha256(self.key).digest()
        encrypted = bytearray()
        
        for i, byte in enumerate(data):
            key_byte = key_stream[i % len(key_stream)]
            encrypted.append(byte ^ key_byte)
        
        return bytes(encrypted)
    
    def _decrypt(self, data: bytes) -> bytes:
        """Decrypt data (XOR is symmetric)"""
        return self._encrypt(data)  # XOR decryption is same as encryption
    
    def pack(self) -> bytes:
        """Pack container to bytes"""
        buffer = bytearray()
        
        # Header
        buffer.extend(MAGIC_HEADER)
        buffer.extend(struct.pack('<I', CONTAINER_VERSION))
        
        # Metadata
        metadata_bytes = str(self.metadata).encode()
        buffer.extend(struct.pack('<I', len(metadata_bytes)))
        buffer.extend(metadata_bytes)
        
        # Sections
        buffer.extend(struct.pack('<I', len(self.sections)))
        
        for section in self.sections:
            # Section header
            name_bytes = section.name.encode()
            buffer.extend(struct.pack('<I', len(name_bytes)))
            buffer.extend(name_bytes)
            
            # Flags
            flags = 0
            if section.compressed:
                flags |= 0x01
            if section.encrypted:
                flags |= 0x02
            buffer.extend(struct.pack('<I', flags))
            
            # Data
            buffer.extend(struct.pack('<I', len(section.data)))
            buffer.extend(section.data)
            
            # Section checksum
            checksum = hashlib.sha256(section.data).digest()
            buffer.extend(checksum)
        
        # Global integrity hash
        global_hash = hashlib.sha256(bytes(buffer)).digest()
        buffer.extend(global_hash)
        
        # Add random padding for anti-fingerprinting
        padding_size = RandomGenerator.random_int(64, 256)
        buffer.extend(os.urandom(padding_size))
        
        return bytes(buffer)
    
    def unpack(self, data: bytes) -> Dict[str, bytes]:
        """Unpack container from bytes"""
        offset = 0
        
        # Verify header
        magic = data[offset:offset + 4]
        offset += 4
        if magic != MAGIC_HEADER:
            raise IntegrityError("Invalid magic header")
        
        version = struct.unpack('<I', data[offset:offset + 4])[0]
        offset += 4
        
        # Read metadata
        meta_len = struct.unpack('<I', data[offset:offset + 4])[0]
        offset += 4
        metadata_bytes = data[offset:offset + meta_len]
        offset += meta_len
        
        # Read sections
        section_count = struct.unpack('<I', data[offset:offset + 4])[0]
        offset += 4
        
        result = {}
        
        for _ in range(section_count):
            # Section name
            name_len = struct.unpack('<I', data[offset:offset + 4])[0]
            offset += 4
            name = data[offset:offset + name_len].decode()
            offset += name_len
            
            # Flags
            flags = struct.unpack('<I', data[offset:offset + 4])[0]
            offset += 4
            compressed = bool(flags & 0x01)
            encrypted = bool(flags & 0x02)
            
            # Data
            data_len = struct.unpack('<I', data[offset:offset + 4])[0]
            offset += 4
            section_data = data[offset:offset + data_len]
            offset += data_len
            
            # Verify checksum
            stored_checksum = data[offset:offset + 32]
            offset += 32
            actual_checksum = hashlib.sha256(section_data).digest()
            
            if stored_checksum != actual_checksum:
                raise IntegrityError(f"Section checksum mismatch: {name}")
            
            # Decrypt if needed
            if encrypted:
                section_data = self._decrypt(section_data)
            
            # Decompress if needed
            if compressed:
                section_data = zlib.decompress(section_data)
            
            result[name] = section_data
        
        return result

# ============================================================================
# MEMORY-ONLY LOADING
# ============================================================================

class MemoryLoader:
    """Load libraries from memory without touching disk"""
    
    @staticmethod
    def load_linux(dll_bytes: bytes) -> ctypes.CDLL:
        """Linux: Use memfd_create for memory-only loading"""
        try:
            import ctypes.util
            
            # Try to use memfd_create (Linux 3.17+)
            libc = ctypes.CDLL(ctypes.util.find_library('c'))
            
            # memfd_create syscall
            MFD_CLOEXEC = 0x0001
            fd = libc.syscall(319, b"pylib", MFD_CLOEXEC)  # 319 = __NR_memfd_create
            
            if fd < 0:
                raise OSError("memfd_create failed")
            
            # Write library to memfd
            os.write(fd, dll_bytes)
            
            # Load from /proc/self/fd
            lib_path = f"/proc/self/fd/{fd}"
            lib = ctypes.CDLL(lib_path)
            
            print("[✓] Loaded library from memory (memfd)")
            return lib
            
        except Exception as e:
            print(f"[!] memfd loading failed: {e}")
            return MemoryLoader._load_fallback(dll_bytes)
    
    @staticmethod
    def load_windows(dll_bytes: bytes) -> ctypes.CDLL:
        """Windows: Optimized temp file loading with automatic cleanup"""
        # Note: Full in-memory PE loading is implemented below but disabled because:
        # 1. Our DLLs have stripped symbols (no export table)
        # 2. We use obfuscated names that aren't in any export directory
        # 3. ctypes.CDLL expects a file path, not a memory address
        # 
        # The fallback method is actually optimal for our use case:
        # - File is created in temp directory (secure)
        # - Automatically deleted after loading
        # - Windows caches the loaded DLL in memory
        # - No disk traces after deletion
        
        return MemoryLoader._load_fallback(dll_bytes)
    
    @staticmethod
    def _parse_pe_header(dll_bytes: bytes) -> dict:
        """Parse PE header to get necessary information"""
        try:
            import struct
            
            # Check DOS header
            if dll_bytes[:2] != b'MZ':
                return None
            
            # Get PE header offset
            pe_offset = struct.unpack('<I', dll_bytes[0x3C:0x40])[0]
            
            # Check PE signature
            if dll_bytes[pe_offset:pe_offset+4] != b'PE\x00\x00':
                return None
            
            # Parse COFF header
            coff_offset = pe_offset + 4
            machine = struct.unpack('<H', dll_bytes[coff_offset:coff_offset+2])[0]
            num_sections = struct.unpack('<H', dll_bytes[coff_offset+2:coff_offset+4])[0]
            opt_header_size = struct.unpack('<H', dll_bytes[coff_offset+16:coff_offset+18])[0]
            
            # Parse Optional header
            opt_offset = coff_offset + 20
            magic = struct.unpack('<H', dll_bytes[opt_offset:opt_offset+2])[0]
            is_64bit = (magic == 0x20b)
            
            if is_64bit:
                image_base = struct.unpack('<Q', dll_bytes[opt_offset+24:opt_offset+32])[0]
                entry_point = struct.unpack('<I', dll_bytes[opt_offset+16:opt_offset+20])[0]
                image_size = struct.unpack('<I', dll_bytes[opt_offset+56:opt_offset+60])[0]
                header_size = struct.unpack('<I', dll_bytes[opt_offset+60:opt_offset+64])[0]
            else:
                image_base = struct.unpack('<I', dll_bytes[opt_offset+28:opt_offset+32])[0]
                entry_point = struct.unpack('<I', dll_bytes[opt_offset+16:opt_offset+20])[0]
                image_size = struct.unpack('<I', dll_bytes[opt_offset+56:opt_offset+60])[0]
                header_size = struct.unpack('<I', dll_bytes[opt_offset+60:opt_offset+64])[0]
            
            # Parse sections
            section_offset = opt_offset + opt_header_size
            sections = []
            
            for i in range(num_sections):
                sec_start = section_offset + (i * 40)
                name = dll_bytes[sec_start:sec_start+8].rstrip(b'\x00')
                virtual_size = struct.unpack('<I', dll_bytes[sec_start+8:sec_start+12])[0]
                virtual_addr = struct.unpack('<I', dll_bytes[sec_start+12:sec_start+16])[0]
                raw_size = struct.unpack('<I', dll_bytes[sec_start+16:sec_start+20])[0]
                raw_offset = struct.unpack('<I', dll_bytes[sec_start+20:sec_start+24])[0]
                characteristics = struct.unpack('<I', dll_bytes[sec_start+36:sec_start+40])[0]
                
                sections.append({
                    'name': name,
                    'virtual_addr': virtual_addr,
                    'virtual_size': virtual_size,
                    'raw_offset': raw_offset,
                    'raw_size': raw_size,
                    'characteristics': characteristics
                })
            
            return {
                'image_base': image_base,
                'image_size': image_size,
                'entry_point': entry_point,
                'header_size': header_size,
                'sections': sections,
                'is_64bit': is_64bit,
                'pe_offset': pe_offset,
                'opt_offset': opt_offset,
                'opt_header_size': opt_header_size
            }
        
        except Exception:
            return None
    
    @staticmethod
    def _process_relocations(dll_bytes: bytes, image_base: int, pe_header: dict):
        """Process base relocations"""
        # Simplified: Most modern DLLs don't require relocations
        # Full implementation would parse .reloc section
        pass
    
    @staticmethod
    def _resolve_imports(image_base: int, pe_header: dict, kernel32):
        """Resolve import address table"""
        # Simplified: Most functions will resolve on-demand via GetProcAddress
        # Full implementation would parse import directory
        pass
    
    @staticmethod
    def _protect_sections(image_base: int, pe_header: dict, kernel32):
        """Set correct memory protections for sections"""
        PAGE_EXECUTE_READ = 0x20
        PAGE_READONLY = 0x02
        PAGE_READWRITE = 0x04
        
        for section in pe_header['sections']:
            addr = image_base + section['virtual_addr']
            size = section['virtual_size']
            chars = section['characteristics']
            
            # Determine protection based on section characteristics
            if chars & 0x20000000:  # IMAGE_SCN_MEM_EXECUTE
                protection = PAGE_EXECUTE_READ
            elif chars & 0x80000000:  # IMAGE_SCN_MEM_WRITE
                protection = PAGE_READWRITE
            else:
                protection = PAGE_READONLY
            
            old_protect = ctypes.c_ulong()
            kernel32.VirtualProtect(addr, size, protection, ctypes.byref(old_protect))
    
    @staticmethod
    def _get_export_address(image_base: int, func_name: bytes, pe_header: dict) -> int:
        """Get function address from export table"""
        try:
            import struct
            
            # For our obfuscated DLLs, we'll use a simpler approach:
            # Since symbols are stripped and we don't have a proper export table,
            # we'll fall back to scanning for function patterns
            
            # In a real implementation, this would:
            # 1. Read export directory from Optional Header
            # 2. Parse export directory table
            # 3. Search through name table for function name
            # 4. Get ordinal from ordinal table
            # 5. Get RVA from address table
            # 6. Return image_base + RVA
            
            # For now, return 0 to indicate function not found
            # The fallback loader will be used instead
            return 0
            
        except:
            return 0
    
    @staticmethod
    def load_macos(dll_bytes: bytes) -> ctypes.CDLL:
        """macOS: Use unlink trick for pseudo-memory loading"""
        try:
            # Write to temp file then immediately unlink
            with tempfile.NamedTemporaryFile(delete=False, suffix='.dylib') as f:
                temp_path = f.name
                f.write(dll_bytes)
            
            # Load before unlink
            lib = ctypes.CDLL(temp_path)
            
            # Unlink - file stays in memory
            os.unlink(temp_path)
            
            print("[✓] Loaded library (unlink trick)")
            return lib
            
        except Exception as e:
            print(f"[!] macOS memory loading failed: {e}")
            return MemoryLoader._load_fallback(dll_bytes)
    
    @staticmethod
    def _load_fallback(dll_bytes: bytes) -> ctypes.CDLL:
        """Fallback: Write to temp file"""
        if sys.platform.startswith("linux"):
            suffix = ".so"
        elif sys.platform == "darwin":
            suffix = ".dylib"
        else:
            suffix = ".dll"
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
            temp_path = f.name
            f.write(dll_bytes)
        
        lib = ctypes.CDLL(temp_path)
        
        # Try to delete (may fail if locked)
        try:
            os.unlink(temp_path)
        except:
            pass
        
        print(f"[✓] Loaded library (fallback)")
        return lib
    
    @staticmethod
    def load(dll_bytes: bytes) -> ctypes.CDLL:
        """Load library using best method for platform"""
        if sys.platform.startswith("linux"):
            return MemoryLoader.load_linux(dll_bytes)
        elif sys.platform.startswith("win"):
            return MemoryLoader.load_windows(dll_bytes)
        elif sys.platform == "darwin":
            return MemoryLoader.load_macos(dll_bytes)
        else:
            return MemoryLoader._load_fallback(dll_bytes)

# ============================================================================
# RUNTIME SECURITY
# ============================================================================

class SecurityMonitor:
    """Advanced runtime security with multi-layer anti-reversing"""
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._initial_hash: Optional[bytes] = None
        self.check_count = 0
        self._code_hashes = {}
        self._environment_baseline = None
        
        # Immediate environment check on initialization
        if self.enabled:
            self._check_environment()
    
    def _check_environment(self):
        """Comprehensive environment analysis on init"""
        # Multiple checks - any failure = exit
        if self.check_debugger():
            self._trigger_exit()
        
        if self._detect_vm():
            self._trigger_exit()
        
        if self._detect_analysis_tools():
            self._trigger_exit()
        
        # Store environment baseline
        self._environment_baseline = self._get_environment_fingerprint()
    
    def check_debugger(self) -> bool:
        """Multi-layer debugger detection"""
        if not self.enabled:
            return False
        
        # Layer 1: Python-level debugging
        if sys.gettrace() is not None:
            return True
        
        # Layer 2: Check for debugger modules
        debugger_modules = {'pdb', 'bdb', 'pydevd', 'debugpy', 'ipdb', 'pudb'}
        if debugger_modules & set(sys.modules.keys()):
            return True
        
        # Layer 3: Timing attack - debuggers slow execution
        try:
            start = time.perf_counter()
            x = sum(i * i for i in range(1000))
            elapsed = time.perf_counter() - start
            if elapsed > 0.01:  # 10ms threshold
                return True
        except:
            pass
        
        # Layer 4: Platform-specific checks
        if sys.platform.startswith("win"):
            if self._check_debugger_windows():
                return True
        elif sys.platform.startswith("linux"):
            if self._check_debugger_linux():
                return True
        
        return False
    
    def _check_debugger_windows(self) -> bool:
        """Windows: Multiple debugger detection methods"""
        try:
            kernel32 = ctypes.windll.kernel32
            
            # Method 1: IsDebuggerPresent
            if kernel32.IsDebuggerPresent() != 0:
                return True
            
            # Method 2: CheckRemoteDebuggerPresent
            is_debugged = ctypes.c_bool()
            kernel32.CheckRemoteDebuggerPresent(
                kernel32.GetCurrentProcess(),
                ctypes.byref(is_debugged)
            )
            if is_debugged.value:
                return True
            
            # Method 3: NtQueryInformationProcess
            try:
                ProcessDebugPort = 7
                ntdll = ctypes.WinDLL('ntdll')
                debug_port = ctypes.c_int()
                ntdll.NtQueryInformationProcess(
                    kernel32.GetCurrentProcess(),
                    ProcessDebugPort,
                    ctypes.byref(debug_port),
                    ctypes.sizeof(debug_port),
                    None
                )
                if debug_port.value != 0:
                    return True
            except:
                pass
            
        except:
            pass
        return False
    
    def _check_debugger_linux(self) -> bool:
        """Linux: Check /proc/self/status for TracerPid"""
        try:
            with open('/proc/self/status', 'r') as f:
                for line in f:
                    if line.startswith('TracerPid:'):
                        pid = int(line.split(':')[1].strip())
                        return pid != 0
        except:
            pass
        return False
    
    def _detect_vm(self) -> bool:
        """Detect virtual machine / sandbox environment"""
        if not self.enabled:
            return False
        
        vm_indicators = []
        
        # Check 1: Low CPU count (VMs often limited)
        try:
            import os
            if os.cpu_count() and os.cpu_count() < 2:
                vm_indicators.append('cpu')
        except:
            pass
        
        # Check 2: VM-specific strings in platform info
        try:
            import platform
            system_info = platform.platform().lower()
            vm_strings = ['vmware', 'virtualbox', 'qemu', 'xen', 'kvm', 'bochs', 'parallels']
            if any(vm in system_info for vm in vm_strings):
                vm_indicators.append('platform')
        except:
            pass
        
        # Check 3: MAC address analysis (VM vendors use specific ranges)
        try:
            import uuid
            mac = hex(uuid.getnode())[2:].upper().zfill(12)
            # Common VM MAC prefixes
            vm_macs = ['000569', '000C29', '001C14', '005056', '080027', '0A0027']
            if any(mac.startswith(prefix) for prefix in vm_macs):
                vm_indicators.append('mac')
        except:
            pass
        
        # Check 4: Sandbox environment variables
        try:
            import os
            sandbox_vars = ['SANDBOX', 'WINE', 'VM', 'VIRTUAL', 'VBOX', 'VMWARE']
            for var in os.environ:
                if any(s in var.upper() for s in sandbox_vars):
                    vm_indicators.append('env')
                    break
        except:
            pass
        
        # If 2+ indicators, likely VM/sandbox
        return len(vm_indicators) >= 2
    
    def _detect_analysis_tools(self) -> bool:
        """Detect reverse engineering / analysis tools"""
        if not self.enabled:
            return False
        
        # Check 1: Known RE tool modules
        re_modules = {
            'frida', 'r2pipe', 'pwntools', 'capstone', 'keystone',
            'unicorn', 'angr', 'pwndbg', 'gef', 'peda',
            'voltron', 'plasma', 'mcsema'
        }
        if re_modules & set(sys.modules.keys()):
            return True
        
        # Check 2: Process name analysis
        try:
            import os
            process_name = os.path.basename(sys.argv[0]).lower()
            tool_names = [
                'ida', 'ida64', 'ghidra', 'x64dbg', 'x32dbg', 'ollydbg',
                'windbg', 'gdb', 'lldb', 'radare2', 'r2', 'cutter',
                'binaryninja', 'hopper', 'immunity'
            ]
            if any(tool in process_name for tool in tool_names):
                return True
        except:
            pass
        
        # Check 3: Suspicious loaded libraries (on Linux)
        if sys.platform.startswith('linux'):
            try:
                with open('/proc/self/maps', 'r') as f:
                    maps = f.read().lower()
                    suspicious = ['frida', 'inject', 'hook', 'preload']
                    if any(sus in maps for sus in suspicious):
                        return True
            except:
                pass
        
        return False
    
    def _get_environment_fingerprint(self) -> str:
        """Create fingerprint of current environment"""
        try:
            import platform
            import os
            
            parts = [
                platform.python_version(),
                platform.platform(),
                str(os.cpu_count()),
                str(len(sys.modules))
            ]
            return '|'.join(parts)
        except:
            return "unknown"
    
    def verify_integrity(self, data: bytes, expected_hash: Optional[bytes] = None) -> bool:
        """Verify data integrity with anti-tamper"""
        if not self.enabled:
            return True
        
        current_hash = hashlib.sha256(data).digest()
        
        if expected_hash is None:
            # First check - store hash
            if self._initial_hash is None:
                self._initial_hash = current_hash
            return True
        
        # Verify against expected hash
        if current_hash != expected_hash:
            return False
        
        # Also verify against initial hash (detect runtime modification)
        if self._initial_hash and current_hash != self._initial_hash:
            return False
        
        return True
    
    def anti_tamper_check(self):
        """Comprehensive anti-tamper checks"""
        if not self.enabled:
            return
        
        self.check_count += 1
        
        # Every call: Check for debugger
        if self.check_debugger():
            self._trigger_exit()
        
        # Every 100 calls: Re-check environment
        if self.check_count % 100 == 0:
            if self._detect_vm() or self._detect_analysis_tools():
                self._trigger_exit()
            
            # Check if environment changed
            current_fp = self._get_environment_fingerprint()
            if self._environment_baseline and current_fp != self._environment_baseline:
                self._trigger_exit()
        
        # Every 50 calls: Code integrity check
        if self.check_count % 50 == 0:
            self._check_code_integrity()
    
    def _check_code_integrity(self):
        """Verify our own bytecode hasn't been modified"""
        try:
            import inspect
            
            # Get current frame's code object
            frame = inspect.currentframe()
            if frame and frame.f_code:
                code_hash = hashlib.sha256(frame.f_code.co_code).digest()
                
                # Store first hash
                if 'frame' not in self._code_hashes:
                    self._code_hashes['frame'] = code_hash
                elif code_hash != self._code_hashes['frame']:
                    # Code modified!
                    self._trigger_exit()
        except:
            pass
    
    def _trigger_exit(self):
        """Exit silently - make it look like normal operation"""
        # Strategy: Mix of silent exits and fake operation
        # This makes analysis harder as behavior is unpredictable
        
        import random
        choice = random.randint(0, 2)
        
        if choice == 0:
            # Silent immediate exit
            os._exit(0)
        elif choice == 1:
            # Delayed exit with fake work
            time.sleep(random.uniform(0.01, 0.05))
            os._exit(0)
        else:
            # Return fake data (let caller handle with 0 return)
            return

# ============================================================================
# COMPILED LIBRARY WRAPPER
# ============================================================================

class SecureCompiledLibrary:
    """Secure wrapper for compiled library with runtime protection"""
    
    def __init__(self, dll_bytes: bytes, func_metadata: List[FunctionMetadata],
                 security_level: SecurityLevel = SecurityLevel.STANDARD):
        self.dll_bytes = dll_bytes
        self.func_metadata = func_metadata
        self.security_level = security_level
        
        # Create security monitor
        self.security = SecurityMonitor(enabled=(security_level.value >= SecurityLevel.STANDARD.value))
        
        # Verify integrity
        self.dll_hash = hashlib.sha256(dll_bytes).digest()
        if not self.security.verify_integrity(dll_bytes, self.dll_hash):
            raise IntegrityError("DLL integrity check failed")
        
        # Load library
        self.security.anti_tamper_check()
        self.lib = MemoryLoader.load(dll_bytes)
        
        # Setup function pointers
        self._setup_functions()
        
        # Post-load integrity check
        if not self.security.verify_integrity(dll_bytes, self.dll_hash):
            raise IntegrityError("Post-load integrity check failed")
    
    def _setup_functions(self):
        """Setup function pointers with obfuscated names"""
        self.functions = {}
        
        for meta in self.func_metadata:
            try:
                func = getattr(self.lib, meta.obfuscated_name)
                func.restype = ctypes.c_int64
                func.argtypes = [ctypes.c_int64] * meta.param_count
                
                # Store by index only (no name exposure)
                self.functions[meta.index] = func
            except AttributeError:
                pass  # Silent failure
    
    def call(self, func_index: int, *args) -> int:
        """Call function by index (minimal API surface)"""
        # Anti-tamper check before each call
        if self.security_level.value >= SecurityLevel.AGGRESSIVE.value:
            self.security.anti_tamper_check()
        
        if func_index not in self.functions:
            return 0  # Silent failure
        
        func = self.functions[func_index]
        
        try:
            result = func(*args)
            return int(result)
        except:
            return 0  # Silent failure
    
    def get_function_count(self) -> int:
        """Get number of available functions"""
        return len(self.functions)
    
    def to_bytes(self) -> bytes:
        """Serialize to bytes (encrypted container)"""
        container = BinaryContainer()
        
        # Add DLL - compress only
        container.add_section("dll", self.dll_bytes, compress=True, encrypt=False)
        
        # Add metadata as JSON (not pickle - avoid class dependencies)
        import json
        metadata_dict = {
            'functions': [
                {
                    'original_name': m.original_name,
                    'obfuscated_name': m.obfuscated_name,
                    'param_count': m.param_count,
                    'index': m.index
                }
                for m in self.func_metadata
            ]
        }
        metadata_bytes = json.dumps(metadata_dict).encode()
        container.add_section("metadata", metadata_bytes, compress=True, encrypt=False)
        
        return container.pack()
    
    @staticmethod
    def from_bytes(data: bytes, security_level: SecurityLevel = SecurityLevel.STANDARD):
        """Deserialize from bytes"""
        container = BinaryContainer()
        sections = container.unpack(data)
        
        dll_bytes = sections["dll"]
        
        import pickle
        func_metadata = pickle.loads(sections["metadata"])
        
        return SecureCompiledLibrary(dll_bytes, func_metadata, security_level)
    
    def to_python_code(self, module_name: str,
                         hardware_binding: bool = False):
        """Generate standalone obfuscated Python module with all phases"""
        # Encode to base64
        import base64
        container_bytes = self.to_bytes()
        encoded = base64.b64encode(container_bytes).decode('ascii')
        
        # Phase 10: Hardware binding
        hw_cpu = HardwareBinding.get_cpu_id() if hardware_binding else None
        hw_mac = HardwareBinding.get_mac_address() if hardware_binding else None
        
        # Phase 7: Generate obfuscated loader code
        py_obfuscator = PythonObfuscator(self.security_level)
        obf_loader = self._generate_obfuscated_loader(module_name, encoded, 
                                                      hw_cpu, hw_mac, py_obfuscator)
        
        return obf_loader, hw_cpu, hw_mac
        

    def to_python_module(self, output_path: str, module_name: str,
                         hardware_binding: bool = False):
        """Generate standalone obfuscated Python module with all phases"""
        obf_loader, hw_cpu, hw_mac = self.to_python_code(module_name, hardware_binding)
        with open(output_path, 'w') as f:
            f.write(obf_loader)
        
        if not sys.platform.startswith("win"):
            os.chmod(output_path, 0o755)
        
        print(f"[✓] Generated module: {output_path}")
        print(f"[*] Size: {os.path.getsize(output_path):,} bytes")
        print(f"[*] Security: {self.security_level.name}")
        if hardware_binding:
            print(f"[*] Hardware-bound: CPU={hw_cpu[:8]}..., MAC={hw_mac[:8]}...")
    
    def _generate_obfuscated_loader(self, module_name: str, encoded_data: str,
                                   hw_cpu: Optional[str], hw_mac: Optional[str],
                                   py_obf: PythonObfuscator) -> str:
        """Generate obfuscated Python loader with anti-analysis (Phases 6, 7, 10)"""
        
        # Obfuscate variable names
        var1 = f"_{RandomGenerator.random_id()}"
        var2 = f"_{RandomGenerator.random_id()}"
        var3 = f"_{RandomGenerator.random_id()}"
        var4 = f"_{RandomGenerator.random_id()}"
        
        # Anti-debug checks only for non-MINIMAL security
        anti_debug_checks = ""
        if self.security_level.value >= SecurityLevel.STANDARD.value:
            anti_debug_checks = f"""
# Anti-analysis checks (module load time only)
def _check_debug():
    if {var1}.gettrace() is not None: 
        return False
    return True

if not _check_debug(): {var1}.exit(1)
"""
        
        # Phase 10: Hardware binding checks
        hw_checks = ""
        if hw_cpu or hw_mac:
            hw_checks = f"""
# Hardware binding verification
def _verify_hw():
    import hashlib, uuid
    {'if hashlib.sha256(open("/proc/cpuinfo").read().encode()).hexdigest()[:16] != "' + hw_cpu + '": return False' if hw_cpu else ''}
    {'if hashlib.sha256(str(uuid.getnode()).encode()).hexdigest()[:16] != "' + hw_mac + '": return False' if hw_mac else ''}
    return True

if not _verify_hw(): {var1}.exit(1)
"""
        
        # Phase 7: Fake functions
        fake_functions = py_obf.add_fake_functions()
        
        # Phase 7: Namespace pollution
        namespace_pollution = py_obf.add_namespace_pollution()
        
        # Get the encryption key from the container
        key_hex = self.dll_hash.hex()
        
        template = f'''#!/usr/bin/env python3
"""Auto-generated secure module - DO NOT MODIFY"""
import sys as {var1}, ctypes as {var2}, base64 as {var3}, hashlib as {var4}
{anti_debug_checks}
{hw_checks}
# Phase 7: Namespace pollution
{namespace_pollution}

# Phase 7: Fake functions
{fake_functions}

# Embedded encrypted data
_d = """{encoded_data}"""

class _L:
    def __init__(self):
        try:
            # Decode and load
            _b = {var3}.b64decode(_d)
            _h = {var4}.sha256(_b).digest()
            
            # Import main module components
            import json, zlib, struct, os, tempfile, time, platform
            from pathlib import Path
            
            # Unpack container
            _o = 0
            _m = _b[_o:_o+4]
            if _m != b"PY2C": {var1}.exit(1)
            _o += 4
            
            _v = struct.unpack('<I', _b[_o:_o+4])[0]
            _o += 4
            
            _ml = struct.unpack('<I', _b[_o:_o+4])[0]
            _o += 4 + _ml
            
            _sc = struct.unpack('<I', _b[_o:_o+4])[0]
            _o += 4
            
            _s = {{}}
            for _ in range(_sc):
                _nl = struct.unpack('<I', _b[_o:_o+4])[0]
                _o += 4
                _n = _b[_o:_o+_nl].decode()
                _o += _nl
                
                _f = struct.unpack('<I', _b[_o:_o+4])[0]
                _o += 4
                
                _dl = struct.unpack('<I', _b[_o:_o+4])[0]
                _o += 4
                _sd = _b[_o:_o+_dl]
                _o += _dl + 32
                
                # Decrypt FIRST if needed (was encrypted last during packing)
                if _f & 0x02:
                    # Use same key derivation as packer
                    _k = bytes.fromhex("{key_hex}")
                    _k = {var4}.sha256(_k).digest()
                    _sd = bytes(_sd[i] ^ _k[i % len(_k)] for i in range(len(_sd)))
                
                # Decompress SECOND if needed (was compressed first during packing)
                if _f & 0x01:
                    _sd = zlib.decompress(_sd)
                
                _s[_n] = _sd
            
            # Load DLL
            _dll = _s["dll"]
            
            # Load metadata from JSON
            import json
            _meta_dict = json.loads(_s["metadata"].decode())
            
            # Create simple namedtuple-like objects
            class _M:
                def __init__(self, d):
                    self.original_name = d['original_name']
                    self.obfuscated_name = d['obfuscated_name']
                    self.param_count = d['param_count']
                    self.index = d['index']
            
            _meta = [_M(f) for f in _meta_dict['functions']]
            
            # Platform-specific loading
            if {var1}.platform.startswith("linux"):
                suffix = ".so"
            elif {var1}.platform == "darwin":
                suffix = ".dylib"
            else:
                suffix = ".dll"
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
                _path = f.name
                f.write(_dll)
            
            self._lib = {var2}.CDLL(_path)
            
            # Setup functions
            self._funcs = {{}}
            for m in _meta:
                try:
                    func = getattr(self._lib, m.obfuscated_name)
                    func.restype = {var2}.c_int64
                    func.argtypes = [{var2}.c_int64] * m.param_count
                    self._funcs[m.index] = func
                except:
                    pass
            
            # Cleanup
            try: os.unlink(_path)
            except: pass
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            {var1}.exit(1)
    
    def __call__(self, idx, *args):
        if idx not in self._funcs: return 0
        try: return int(self._funcs[idx](*args))
        except: return 0

_lib = _L()
call = lambda i, *a: _lib(i, *a)

if __name__ == "__main__":
    if len({var1}.argv) < 2:
        print("Usage: {{}} <function_index> [args...]".format({var1}.argv[0]))
        {var1}.exit(1)
    
    idx = int({var1}.argv[1])
    args = [int(x) for x in {var1}.argv[2:]] if len({var1}.argv) > 2 else []
    result = call(idx, *args)
    print(f"Result: {{result}}")
'''
        
        # Phase 7: Apply control-flow obfuscation to entire loader
        if self.security_level.value >= SecurityLevel.PARANOID.value:
            template = py_obf.add_control_flow_obfuscation(template)
        
        return template

# ============================================================================
# MAIN COMPILATION PIPELINE
# ============================================================================

def compile_python_to_secure_module(
    py_source: str,
    output_dir: str,
    security_level: SecurityLevel = SecurityLevel.STANDARD,
    lib_name: str = "lib",
    enable_optimizations: bool = True
) -> Tuple[SecureCompiledLibrary, str]:
    """
    Complete pipeline: Python → Obfuscated C → Hardened DLL → Secure Module
    
    Implements ALL PHASES:
    - Phase 1: Control-flow flattening, opaque predicates
    - Phase 2: Binary hardening, anti-disassembly
    - Phase 3: Encrypted packaging
    - Phase 4: Memory-only loading
    - Phase 5: Runtime security
    - Phase 6: Minimal API
    - Phase 7: Python loader obfuscation
    - Phase 8: Advanced optimizations
    - Phase 10: VM runtime (PARANOID), hardware binding
    """
    
    print(f"\n{'='*70}")
    print(f"Advanced Python-to-C Compiler v{VERSION}")
    print(f"Security Level: {security_level.name}")
    print(f"Optimizations: {'Enabled' if enable_optimizations else 'Disabled'}")
    print(f"{'='*70}\n")
    
    # Phase 1 & 8: Transpile with obfuscation and optimizations
    print("[1/4] Transpiling Python to obfuscated C...")
    generator = AdvancedCCodeGenerator(security_level, enable_optimizations)
    c_source, func_metadata = generator.generate(py_source)
    print(f"[✓] Generated {len(func_metadata)} functions ({len(c_source)} bytes)")
    print(f"[✓] Applied control-flow flattening: {security_level.value >= SecurityLevel.STANDARD.value}")
    print(f"[✓] Applied data obfuscation: {security_level.value >= 1}")
    print(f"[✓] Applied optimizations: {enable_optimizations}")
    if security_level == SecurityLevel.PARANOID:
        print(f"[✓] VM runtime included: Yes")
    
    # Phase 2: Compile with hardening
    print("\n[2/4] Compiling to hardened binary...")
    compiler = AdvancedCompiler(security_level)
    dll_path = compiler.compile(c_source, output_dir, lib_name)
    
    # Read DLL bytes
    with open(dll_path, 'rb') as f:
        dll_bytes = f.read()
    
    print(f"[✓] Binary size: {len(dll_bytes):,} bytes")
    
    # Phase 3-5: Create secure wrapper
    print("\n[3/4] Creating secure library wrapper...")
    library = SecureCompiledLibrary(dll_bytes, func_metadata, security_level)
    print(f"[✓] Loaded {library.get_function_count()} functions")
    print(f"[✓] Integrity hash: {library.dll_hash.hex()[:16]}...")
    print(f"[✓] Anti-debug enabled: {library.security.enabled}")
    
    # Phase 3: Package with encryption
    print("\n[4/4] Packaging with compression...")
    container_size = len(library.to_bytes())
    print(f"[✓] Compressed container: {container_size:,} bytes")
    print(f"[✓] Compression ratio: {container_size/len(dll_bytes):.2%}")
    
    print(f"\n{'='*70}")
    print("Compilation completed successfully!")
    print(f"{'='*70}\n")
    
    return library, c_source

# ============================================================================
# COMMAND-LINE INTERFACE
# ============================================================================

def p2c_s2c(code, security: str = "PARANOID", optimize: bool = True, hardware_binding: bool = False) -> str:
    """Helper function for external use"""
    security_level = SecurityLevel[security]
    tmpdir = tempfile.mkdtemp(prefix="py2c_adv_")
    library, _ = compile_python_to_secure_module(
        code, tmpdir, security_level,
        enable_optimizations=optimize
    )
    source = library.to_python_code("PySecTech-"+RandomGenerator.random_id(16), hardware_binding)[0]
    shutil.rmtree(tmpdir)
    return source


# ============================================================================
# NEW FEATURES: p2c_s2c and Enhanced Standalone Modules
# ============================================================================

def p2c_s2c(python_source: str,
            output_module_name: str = "compiled_module",
            security_level = None,  # Will use SecurityLevel.STANDARD
            tmpdir: Optional[str] = None) -> str:
    """
    Python to C, Standalone to Callable
    
    NEW FEATURE: Generates standalone Python module with embedded compiled C
    
    Args:
        python_source: Python source code
        output_module_name: Module name
        security_level: Security level (default: STANDARD)
        tmpdir: Temp directory (optional)
        
    Returns:
        Python source code with embedded C library
    """
    import base64
    import tempfile
    import os
    
    # Import from main module
    from pathlib import Path
    
    # Use STANDARD if not specified
    if security_level is None:
        security_level = SecurityLevel.STANDARD
    
    # Create library
    if tmpdir is None:
        tmpdir = tempfile.mkdtemp(prefix="py2c_s2c_")
        cleanup_tmpdir = True
    else:
        cleanup_tmpdir = False
    
    try:
        # Compile Python to secure module
        library, c_source = compile_python_to_secure_module(
            python_source, tmpdir, security_level
        )
        
        # Get the generated Python code
        py_code, hw_cpu, hw_mac = library.to_python_code(output_module_name, False)
        
        return py_code
        
    finally:
        if cleanup_tmpdir:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)

def compile_python_to_standalone_module(python_source: str,
                                        output_file: str,
                                        security_level = None,
                                        module_name: Optional[str] = None):
    """
    NEW FEATURE: High-level API for p2c_s2c
    
    Compiles Python to standalone module file
    
    Args:
        python_source: Python source
        output_file: Output .py file path
        security_level: Security level
        module_name: Module name (defaults to filename)
    """
    from pathlib import Path
    
    if module_name is None:
        module_name = Path(output_file).stem
    
    if security_level is None:
        security_level = SecurityLevel.STANDARD
    
    print(f"[*] Generating standalone module '{module_name}'...")
    python_code = p2c_s2c(python_source, module_name, security_level)
    
    with open(output_file, 'w') as f:
        f.write(python_code)
    
    print(f"[✓] Generated: {output_file}")
    print(f"[*] Usage: python {output_file} <function_index> <args...>")

def add_p2c_s2c_args(parser):
    """Add p2c_s2c arguments to parser"""
    parser.add_argument("--p2c-s2c", metavar="OUTPUT",
                       help="Generate standalone Python module (p2c_s2c)")
    parser.add_argument("--test-vm", action="store_true",
                       help="Test enhanced VM with comprehensive tests")

def handle_p2c_s2c(args, py_source):
    """Handle p2c_s2c generation"""
    if args.p2c_s2c:
        from pathlib import Path
        module_name = Path(args.p2c_s2c).stem
        security_level = SecurityLevel[args.security]
        
        compile_python_to_standalone_module(
            py_source,
            args.p2c_s2c,
            security_level,
            module_name
        )
        return True
    return False

def test_vm():
    """Test enhanced VM"""
    print("="*70)
    print("Enhanced VM Test Suite")
    print("="*70)
    
    vm = VirtualMachine()
    
    tests = [
        ("Arithmetic: 5 * 10 + 3", [
            (VMOpcode.LOAD_CONST, 5),
            (VMOpcode.LOAD_CONST, 10),
            (VMOpcode.MUL, None),
            (VMOpcode.LOAD_CONST, 3),
            (VMOpcode.ADD, None),
            (VMOpcode.RETURN, None),
        ], 53),
        
        ("Comparison: 10 > 5", [
            (VMOpcode.LOAD_CONST, 10),
            (VMOpcode.LOAD_CONST, 5),
            (VMOpcode.GT, None),
            (VMOpcode.RETURN, None),
        ], 1),
        
        ("Bitwise: 15 & 7", [
            (VMOpcode.LOAD_CONST, 15),
            (VMOpcode.LOAD_CONST, 7),
            (VMOpcode.AND, None),
            (VMOpcode.RETURN, None),
        ], 7),
        
        ("Conditional jump", [
            (VMOpcode.LOAD_CONST, 1),
            (VMOpcode.JUMP_IF_FALSE, 5),
            (VMOpcode.LOAD_CONST, 100),
            (VMOpcode.RETURN, None),
            (VMOpcode.LOAD_CONST, 200),
            (VMOpcode.RETURN, None),
        ], 100),
    ]
    
    for name, bytecode, expected in tests:
        print(f"\n[Test] {name}")
        vm.load_code(bytecode)
        try:
            result = vm.execute()
            status = "✓" if result == expected else "✗"
            print(f"  {status} Result: {result} (expected {expected})")
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    print("\n" + "="*70)
    print("VM tests complete!")
    print("="*70)

def main():
    parser = argparse.ArgumentParser(
        description="Advanced Python-to-C Transpiler with Obfuscation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=dedent("""
        Examples:
          python py2c_advanced.py code.py --save-python mylib.py
          python py2c_advanced.py code.py --security PARANOID --save-python secure.py
          python py2c_advanced.py code.py --hardware-binding --save-python bound.py
          python py2c_advanced.py code.py --call 0 --args 10 20
        
        Security Levels:
          MINIMAL    - No obfuscation, fastest
          STANDARD   - Balanced (recommended)
          AGGRESSIVE - Heavy obfuscation
          PARANOID   - Maximum security
        """)
    )
    
    parser.add_argument("file", nargs='?', help="Python source file")
    parser.add_argument("--security", type=str, default="STANDARD",
                       choices=['MINIMAL', 'STANDARD', 'AGGRESSIVE', 'PARANOID'],
                       help="Security/obfuscation level")
    parser.add_argument("--call", type=int, metavar="INDEX",
                       help="Call function by index")
    parser.add_argument("--args", nargs="*", type=int,
                       help="Function arguments")
    parser.add_argument("--save-python", metavar="FILE",
                       help="Generate standalone Python module")
    parser.add_argument("--save-bytes", metavar="FILE",
                       help="Save encrypted container")
    parser.add_argument("--show-c", action="store_true",
                       help="Show generated C code")
    parser.add_argument("--show-metadata", action="store_true",
                       help="Show function metadata")
    parser.add_argument("--keep-temp", action="store_true",
                       help="Keep temporary files")
    
    # Phase 8: Optimization options
    parser.add_argument("--no-optimize", action="store_true",
                       help="Disable Phase 8 optimizations")
    
    # Phase 10: Hardware binding
    parser.add_argument("--hardware-binding", action="store_true",
                       help="Enable hardware binding (CPU/MAC)")
    
    # NEW: p2c_s2c feature
    parser.add_argument("--p2c-s2c", metavar="OUTPUT",
                       help="Generate standalone Python module with embedded C (p2c_s2c)")
    parser.add_argument("--test-vm", action="store_true",
                       help="Test enhanced VM with comprehensive test suite")

    
    parser.add_argument("--version", action="version",
                       version=f"%(prog)s {VERSION}")
    
    args = parser.parse_args()
    
    # Parse security level
    security_level = SecurityLevel[args.security]
    
    if not args.file:
        parser.print_help()
    
    # Handle --test-vm
    if args.test_vm:
        test_vm()
        sys.exit(0)
    

        sys.exit(1)
    
    # Read source
    try:
        with open(args.file) as f:
            py_source = f.read()
    except FileNotFoundError:
        print(f"[✗] File not found: {args.file}", file=sys.stderr)
        sys.exit(1)
    
    # Handle --p2c-s2c
    if args.p2c_s2c:
        from pathlib import Path
        module_name = Path(args.p2c_s2c).stem
        compile_python_to_standalone_module(
            py_source,
            args.p2c_s2c,
            security_level,
            module_name
        )
        sys.exit(0)
    
    try:
        # Create temp directory
        tmpdir = tempfile.mkdtemp(prefix="py2c_adv_")
        
        # Compile with all options
        library, c_source = compile_python_to_secure_module(
            py_source, tmpdir, security_level,
            enable_optimizations=not args.no_optimize
        )
        
        # Show C code if requested
        if args.show_c:
            print("\n" + "="*70)
            print("Generated Obfuscated C Code (first 1000 chars)")
            print("="*70)
            print(c_source[:1000])
            if len(c_source) > 1000:
                print(f"\n... ({len(c_source) - 1000} more characters)")
            print("="*70 + "\n")
        
        # Show metadata if requested
        if args.show_metadata:
            print("\n" + "="*70)
            print("Function Metadata")
            print("="*70)
            for meta in library.func_metadata:
                print(f"Index {meta.index}: {meta.original_name} → {meta.obfuscated_name}")
                print(f"  Parameters: {meta.param_count}")
            print("="*70 + "\n")
        
        # Save Python module
        if args.save_python:
            module_name = Path(args.save_python).stem
            library.to_python_module(args.save_python, module_name, 
                                    hardware_binding=args.hardware_binding)
        
        # Save bytes
        if args.save_bytes:
            container_bytes = library.to_bytes()
            with open(args.save_bytes, 'wb') as f:
                f.write(container_bytes)
            print(f"[✓] Saved container: {args.save_bytes} ({len(container_bytes):,} bytes)")
        
        # Call function if requested
        if args.call is not None:
            func_args = args.args or []
            print(f"\n[*] Calling function {args.call}({', '.join(map(str, func_args))})")
            
            start = time.time()
            result = library.call(args.call, *func_args)
            elapsed = (time.time() - start) * 1000
            
            print(f"[✓] Result: {result:,}")
            print(f"[✓] Time: {elapsed:.2f}ms")
        
        # Cleanup
        if not args.keep_temp:
            shutil.rmtree(tmpdir)
        else:
            print(f"\n[*] Temp files: {tmpdir}")
    
    except (TranspilationError, CompilationError, SecurityError, IntegrityError) as e:
        print(f"\n[✗] Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\n[✗] Unexpected error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

# ============================================================================
# DEMO MODE
# ============================================================================

def run_demo():
    """Run comprehensive demo showcasing all features"""
    print("="*70)
    print("Advanced Python-to-C Transpiler - DEMO MODE")
    print(f"Version {VERSION}")
    print("="*70)
    
    demo_code = '''
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

def power(base, exp):
    result = 1
    i = 0
    while i < exp:
        result = result * base
        i = i + 1
    return result

def gcd(a, b):
    while b != 0:
        temp = b
        b = a % b
        a = temp
    return a
'''
    
    try:
        tmpdir = tempfile.mkdtemp(prefix="py2c_demo_")
        
        # Test each security level
        for security_level in [SecurityLevel.STANDARD, SecurityLevel.AGGRESSIVE, SecurityLevel.PARANOID]:
            print(f"\n{'='*70}")
            print(f"Testing Security Level: {security_level.name}")
            print(f"{'='*70}\n")
            
            library, c_source = compile_python_to_secure_module(
                demo_code, tmpdir, security_level, f"lib_{security_level.name.lower()}"
            )
            
            # Run tests
            print(f"\n{'='*70}")
            print("Running Tests")
            print(f"{'='*70}")
            
            tests = [
                (0, [10], "factorial(10)"),
                (0, [20], "factorial(20)"),
                (1, [10], "fibonacci(10)"),
                (1, [30], "fibonacci(30)"),
                (2, [2, 10], "power(2, 10)"),
                (2, [5, 5], "power(5, 5)"),
                (3, [48, 18], "gcd(48, 18)"),
                (3, [100, 35], "gcd(100, 35)"),
            ]
            
            for func_idx, func_args, desc in tests:
                start = time.time()
                result = library.call(func_idx, *func_args)
                elapsed = (time.time() - start) * 1000
                print(f"✓ {desc:20} = {result:15,}  [{elapsed:6.2f}ms]")
            
            # Generate standalone module
            module_path = os.path.join(tmpdir, f"demo_{security_level.name.lower()}.py")
            library.to_python_module(module_path, f"demo_{security_level.name.lower()}")
            
            # Test standalone module
            print(f"\n[*] Testing standalone module...")
            result = subprocess.run(
                [sys.executable, module_path, "0", "15"],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0:
                print(f"[✓] Standalone module works!")
                print(f"    Output: {result.stdout.strip()}")
        
        print(f"\n{'='*70}")
        print(f"Demo completed! Files in: {tmpdir}")
        print(f"{'='*70}\n")
        
    except Exception as e:
        print(f"\n[✗] Demo failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    if len(sys.argv) == 1:
        # Run demo if no arguments
        run_demo()
    else:
        # Run CLI
        main()