from enum import Enum
from typing import List, Tuple
from .base import SecurityLevel
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
        self.enabled = True if security_level and security_level.value >= SecurityLevel.STANDARD.value else False
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

