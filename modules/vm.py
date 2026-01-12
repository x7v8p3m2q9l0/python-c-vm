import ast
import secrets
import hashlib
from typing import List, Tuple, Optional, Dict, Any
from enum import Enum

try:
    from .type import VMOpcodeBase, OpcodeMapping
    from .base import SecurityLevel, CompilationError
    from .utils import RandomGenerator
except ImportError:
    # Standalone mode
    class VMOpcodeBase(Enum):
        LOAD_CONST = 0x01
        LOAD_VAR = 0x02
        STORE_VAR = 0x03
        LOAD_PARAM = 0x04
        ADD = 0x10
        SUB = 0x11
        MUL = 0x12
        DIV = 0x13
        MOD = 0x14
        NEG = 0x15
        AND = 0x20
        OR = 0x21
        XOR = 0x22
        NOT = 0x23
        SHL = 0x24
        SHR = 0x25
        LT = 0x30
        LE = 0x31
        GT = 0x32
        GE = 0x33
        EQ = 0x34
        NE = 0x35
        JUMP = 0x40
        JUMP_IF_FALSE = 0x41
        JUMP_IF_TRUE = 0x42
        CALL = 0x50
        RETURN = 0x51
        DUP = 0x60
        POP = 0x61
        SWAP = 0x62
        NOP = 0xFE
        HALT = 0xFF
    
    class OpcodeMapping:
        def __init__(self):
            self.salt = secrets.token_bytes(32)
            self.forward = {}
            self.reverse = {}
            self._generate_mapping()
            self.checksum = self._compute_checksum()
        
        def _generate_mapping(self):
            base_opcodes = [op.value for op in VMOpcodeBase]
            random_pool = list(range(256))
            secrets.SystemRandom().shuffle(random_pool)
            for i, base_op in enumerate(base_opcodes):
                runtime_op = random_pool[i]
                self.forward[base_op] = runtime_op
                self.reverse[runtime_op] = base_op
        
        def _compute_checksum(self) -> bytes:
            data = b''
            for base_op in sorted(self.forward.keys()):
                runtime_op = self.forward[base_op]
                data += base_op.to_bytes(1, 'little')
                data += runtime_op.to_bytes(1, 'little')
            data += self.salt
            return hashlib.sha256(data).digest()
        
        def map_opcode(self, base_opcode: int) -> int:
            return self.forward.get(base_opcode, base_opcode)
    
    class SecurityLevel(Enum):
        MINIMAL = 0
        STANDARD = 1
        AGGRESSIVE = 2
        PARANOID = 3
    
    class CompilationError(Exception):
        pass

__version__ = "6.2.0-complete-fix"


class VMCompiler:
    """VM Compiler - Returns plain tuples, parameters work correctly"""
    
    def __init__(self, mapping: Optional[OpcodeMapping] = None):
        self.mapping = mapping or OpcodeMapping()
        self.bytecode: List[Tuple[VMOpcodeBase, Optional[int]]] = []
        self.var_map: Dict[str, int] = {}
        self.next_var_id = 0
        self.function_name = ""
        self.param_names: List[str] = []
        self.error_context = []
    
    def compile_function(self, func_def: ast.FunctionDef) -> List[Tuple[VMOpcodeBase, Optional[int]]]:
        """Compile a Python function to VM bytecode"""
        try:
            self.bytecode = []
            self.var_map = {}
            self.next_var_id = 0
            self.function_name = func_def.name
            self.param_names = []
            self.error_context = []
            
            # CRITICAL FIX: Map parameters as variables (not separate)
            # This matches how main.py passes them!
            for i, arg in enumerate(func_def.args.args):
                param_name = arg.arg
                self.param_names.append(param_name)
                # Map parameter to variable ID
                self.var_map[param_name] = i
                self.next_var_id = i + 1
            
            # Load parameters into variables at function start
            for i, param_name in enumerate(self.param_names):
                self.emit(VMOpcodeBase.LOAD_PARAM, i)
                self.emit(VMOpcodeBase.STORE_VAR, i)
            
            # Compile function body
            for stmt in func_def.body:
                self._compile_stmt(stmt)
            
            # Ensure return
            if not self.bytecode or self.bytecode[-1][0] != VMOpcodeBase.RETURN:
                self.emit(VMOpcodeBase.LOAD_CONST, 0)
                self.emit(VMOpcodeBase.RETURN)
            
            return self.bytecode
            
        except Exception as e:
            context = "\n".join(self.error_context) if self.error_context else "unknown"
            raise CompilationError(f"VM compilation failed for {func_def.name}: {e}\nContext: {context}")
    
    def emit(self, opcode: VMOpcodeBase, arg: Optional[int] = None):
        """Emit a VM instruction as plain tuple"""
        self.bytecode.append((opcode, arg))
    
    def _get_var_id(self, name: str) -> int:
        """Get or create variable ID"""
        if name not in self.var_map:
            self.var_map[name] = self.next_var_id
            self.next_var_id += 1
        return self.var_map[name]
    
    def _compile_stmt(self, stmt: ast.stmt):
        """Compile a statement"""
        try:
            if isinstance(stmt, ast.Return):
                self.error_context.append(f"Compiling return statement")
                if stmt.value:
                    self._compile_expr(stmt.value)
                else:
                    self.emit(VMOpcodeBase.LOAD_CONST, 0)
                self.emit(VMOpcodeBase.RETURN)
            
            elif isinstance(stmt, ast.Assign):
                self.error_context.append(f"Compiling assignment")
                if len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
                    self._compile_expr(stmt.value)
                    var_id = self._get_var_id(stmt.targets[0].id)
                    self.emit(VMOpcodeBase.STORE_VAR, var_id)
            
            elif isinstance(stmt, ast.AugAssign):
                self.error_context.append(f"Compiling augmented assignment")
                if isinstance(stmt.target, ast.Name):
                    var_id = self._get_var_id(stmt.target.id)
                    self.emit(VMOpcodeBase.LOAD_VAR, var_id)
                    self._compile_expr(stmt.value)
                    self._compile_binop(stmt.op)
                    self.emit(VMOpcodeBase.STORE_VAR, var_id)
            
            elif isinstance(stmt, ast.If):
                self.error_context.append(f"Compiling if statement")
                self._compile_if(stmt)
            
            elif isinstance(stmt, ast.While):
                self.error_context.append(f"Compiling while loop")
                self._compile_while(stmt)
            
            elif isinstance(stmt, ast.Expr):
                self.error_context.append(f"Compiling expression statement")
                self._compile_expr(stmt.value)
                self.emit(VMOpcodeBase.POP)
            
            elif isinstance(stmt, ast.Pass):
                self.emit(VMOpcodeBase.NOP)
            
            else:
                raise CompilationError(f"Unsupported statement type: {type(stmt).__name__}")
                
        finally:
            if self.error_context:
                self.error_context.pop()
    
    def _compile_expr(self, expr: ast.expr):
        """Compile an expression"""
        try:
            # Handle Constant (Python 3.8+)
            if isinstance(expr, ast.Constant):
                value = expr.value
                if isinstance(value, (int, bool)):
                    self.emit(VMOpcodeBase.LOAD_CONST, int(value))
                elif isinstance(value, float):
                    self.emit(VMOpcodeBase.LOAD_CONST, int(value))
                elif value is None:
                    self.emit(VMOpcodeBase.LOAD_CONST, 0)
                else:
                    raise CompilationError(f"Unsupported constant type: {type(value)}")
            
            elif isinstance(expr, ast.Num):
                self.emit(VMOpcodeBase.LOAD_CONST, int(expr.n))
            
            elif isinstance(expr, ast.NameConstant):
                if expr.value is True:
                    self.emit(VMOpcodeBase.LOAD_CONST, 1)
                elif expr.value is False:
                    self.emit(VMOpcodeBase.LOAD_CONST, 0)
                elif expr.value is None:
                    self.emit(VMOpcodeBase.LOAD_CONST, 0)
            
            elif isinstance(expr, ast.Name):
                var_id = self._get_var_id(expr.id)
                self.emit(VMOpcodeBase.LOAD_VAR, var_id)
            
            elif isinstance(expr, ast.BinOp):
                self._compile_expr(expr.left)
                self._compile_expr(expr.right)
                self._compile_binop(expr.op)
            
            elif isinstance(expr, ast.UnaryOp):
                self._compile_expr(expr.operand)
                if isinstance(expr.op, ast.USub):
                    self.emit(VMOpcodeBase.NEG)
                elif isinstance(expr.op, ast.Not):
                    self.emit(VMOpcodeBase.NOT)
                elif isinstance(expr.op, ast.UAdd):
                    pass
                elif isinstance(expr.op, ast.Invert):
                    self.emit(VMOpcodeBase.NOT)
            
            elif isinstance(expr, ast.Compare):
                if len(expr.ops) == 1 and len(expr.comparators) == 1:
                    self._compile_expr(expr.left)
                    self._compile_expr(expr.comparators[0])
                    self._compile_compare(expr.ops[0])
                else:
                    self._compile_expr(expr.left)
                    for i, (op, comparator) in enumerate(zip(expr.ops, expr.comparators)):
                        self.emit(VMOpcodeBase.DUP)
                        self._compile_expr(comparator)
                        self._compile_compare(op)
                        if i < len(expr.ops) - 1:
                            self.emit(VMOpcodeBase.AND)
            
            elif isinstance(expr, ast.BoolOp):
                if isinstance(expr.op, ast.And):
                    self._compile_expr(expr.values[0])
                    for value in expr.values[1:]:
                        self.emit(VMOpcodeBase.DUP)
                        skip_jump = len(self.bytecode)
                        self.emit(VMOpcodeBase.JUMP_IF_FALSE, 0)
                        self.emit(VMOpcodeBase.POP)
                        self._compile_expr(value)
                        skip_pos = len(self.bytecode)
                        self.bytecode[skip_jump] = (VMOpcodeBase.JUMP_IF_FALSE, skip_pos)
                
                elif isinstance(expr.op, ast.Or):
                    self._compile_expr(expr.values[0])
                    for value in expr.values[1:]:
                        self.emit(VMOpcodeBase.DUP)
                        skip_jump = len(self.bytecode)
                        self.emit(VMOpcodeBase.JUMP_IF_TRUE, 0)
                        self.emit(VMOpcodeBase.POP)
                        self._compile_expr(value)
                        skip_pos = len(self.bytecode)
                        self.bytecode[skip_jump] = (VMOpcodeBase.JUMP_IF_TRUE, skip_pos)
            
            else:
                raise CompilationError(f"Unsupported expression type: {type(expr).__name__}")
                
        except Exception as e:
            raise CompilationError(f"Error compiling expression {ast.dump(expr)}: {e}")
    
    def _compile_binop(self, op: ast.operator):
        """Compile binary operator"""
        op_map = {
            ast.Add: VMOpcodeBase.ADD,
            ast.Sub: VMOpcodeBase.SUB,
            ast.Mult: VMOpcodeBase.MUL,
            ast.Div: VMOpcodeBase.DIV,
            ast.FloorDiv: VMOpcodeBase.DIV,
            ast.Mod: VMOpcodeBase.MOD,
            ast.BitAnd: VMOpcodeBase.AND,
            ast.BitOr: VMOpcodeBase.OR,
            ast.BitXor: VMOpcodeBase.XOR,
            ast.LShift: VMOpcodeBase.SHL,
            ast.RShift: VMOpcodeBase.SHR,
        }
        
        op_type = type(op)
        if op_type in op_map:
            self.emit(op_map[op_type])
        else:
            raise CompilationError(f"Unsupported binary operator: {op_type.__name__}")
    
    def _compile_compare(self, op: ast.cmpop):
        """Compile comparison operator"""
        op_map = {
            ast.Lt: VMOpcodeBase.LT,
            ast.LtE: VMOpcodeBase.LE,
            ast.Gt: VMOpcodeBase.GT,
            ast.GtE: VMOpcodeBase.GE,
            ast.Eq: VMOpcodeBase.EQ,
            ast.NotEq: VMOpcodeBase.NE,
        }
        
        op_type = type(op)
        if op_type in op_map:
            self.emit(op_map[op_type])
        else:
            raise CompilationError(f"Unsupported comparison operator: {op_type.__name__}")
    
    def _compile_if(self, stmt: ast.If):
        """Compile if statement"""
        self._compile_expr(stmt.test)
        
        jump_to_else = len(self.bytecode)
        self.emit(VMOpcodeBase.JUMP_IF_FALSE, 0)
        
        for s in stmt.body:
            self._compile_stmt(s)
        
        if stmt.orelse:
            jump_to_end = len(self.bytecode)
            self.emit(VMOpcodeBase.JUMP, 0)
            
            else_pos = len(self.bytecode)
            self.bytecode[jump_to_else] = (VMOpcodeBase.JUMP_IF_FALSE, else_pos)
            
            for s in stmt.orelse:
                self._compile_stmt(s)
            
            end_pos = len(self.bytecode)
            self.bytecode[jump_to_end] = (VMOpcodeBase.JUMP, end_pos)
        else:
            end_pos = len(self.bytecode)
            self.bytecode[jump_to_else] = (VMOpcodeBase.JUMP_IF_FALSE, end_pos)
    
    def _compile_while(self, stmt: ast.While):
        """Compile while loop"""
        loop_start = len(self.bytecode)
        
        self._compile_expr(stmt.test)
        
        jump_to_end = len(self.bytecode)
        self.emit(VMOpcodeBase.JUMP_IF_FALSE, 0)
        
        for s in stmt.body:
            self._compile_stmt(s)
        
        self.emit(VMOpcodeBase.JUMP, loop_start)
        
        end_pos = len(self.bytecode)
        self.bytecode[jump_to_end] = (VMOpcodeBase.JUMP_IF_FALSE, end_pos)
    
    def disassemble(self) -> str:
        """Disassemble bytecode for debugging"""
        lines = []
        lines.append(f"Function: {self.function_name}")
        lines.append(f"Parameters: {', '.join(self.param_names)}")
        lines.append(f"Variables: {self.var_map}")
        lines.append(f"Instructions: {len(self.bytecode)}")
        lines.append("")
        
        for i, (opcode, arg) in enumerate(self.bytecode):
            if arg is not None:
                lines.append(f"{i:4d}: {opcode.name:20s} {arg}")
            else:
                lines.append(f"{i:4d}: {opcode.name}")
        
        return "\n".join(lines)


def generate_c_runtime(mapping: OpcodeMapping, function_name: str = "vm_execute") -> str:
    """
    Generate C runtime - CRITICAL FIX: Properly decode bytecode format
    The bytecode format from main.py is: [opcode:1byte][arg:4bytes] repeated
    """
    
    # Generate opcode case statements
    opcode_cases = ""
    
    op_handlers = {
        VMOpcodeBase.LOAD_CONST.value: "if (vm->sp < 1024) vm->stack[vm->sp++] = arg;",
        VMOpcodeBase.LOAD_VAR.value: "if (vm->sp < 1024 && arg < 256) vm->stack[vm->sp++] = vm->vars[arg];",
        VMOpcodeBase.STORE_VAR.value: "if (vm->sp > 0 && arg < 256) vm->vars[arg] = vm->stack[--vm->sp];",
        VMOpcodeBase.LOAD_PARAM.value: "if (vm->sp < 1024 && arg < vm->param_count) vm->stack[vm->sp++] = vm->params[arg]; else if (vm->sp < 1024) vm->stack[vm->sp++] = 0;",
        VMOpcodeBase.ADD.value: "if (vm->sp >= 2) { int64_t b = vm->stack[--vm->sp]; int64_t a = vm->stack[--vm->sp]; vm->stack[vm->sp++] = a + b; }",
        VMOpcodeBase.SUB.value: "if (vm->sp >= 2) { int64_t b = vm->stack[--vm->sp]; int64_t a = vm->stack[--vm->sp]; vm->stack[vm->sp++] = a - b; }",
        VMOpcodeBase.MUL.value: "if (vm->sp >= 2) { int64_t b = vm->stack[--vm->sp]; int64_t a = vm->stack[--vm->sp]; vm->stack[vm->sp++] = a * b; }",
        VMOpcodeBase.DIV.value: "if (vm->sp >= 2) { int64_t b = vm->stack[--vm->sp]; int64_t a = vm->stack[--vm->sp]; if (b != 0) vm->stack[vm->sp++] = a / b; else vm->stack[vm->sp++] = 0; }",
        VMOpcodeBase.MOD.value: "if (vm->sp >= 2) { int64_t b = vm->stack[--vm->sp]; int64_t a = vm->stack[--vm->sp]; if (b != 0) vm->stack[vm->sp++] = a % b; else vm->stack[vm->sp++] = 0; }",
        VMOpcodeBase.NEG.value: "if (vm->sp > 0) vm->stack[vm->sp-1] = -vm->stack[vm->sp-1];",
        VMOpcodeBase.AND.value: "if (vm->sp >= 2) { int64_t b = vm->stack[--vm->sp]; int64_t a = vm->stack[--vm->sp]; vm->stack[vm->sp++] = a & b; }",
        VMOpcodeBase.OR.value: "if (vm->sp >= 2) { int64_t b = vm->stack[--vm->sp]; int64_t a = vm->stack[--vm->sp]; vm->stack[vm->sp++] = a | b; }",
        VMOpcodeBase.XOR.value: "if (vm->sp >= 2) { int64_t b = vm->stack[--vm->sp]; int64_t a = vm->stack[--vm->sp]; vm->stack[vm->sp++] = a ^ b; }",
        VMOpcodeBase.NOT.value: "if (vm->sp > 0) vm->stack[vm->sp-1] = ~vm->stack[vm->sp-1];",
        VMOpcodeBase.SHL.value: "if (vm->sp >= 2) { int64_t b = vm->stack[--vm->sp]; int64_t a = vm->stack[--vm->sp]; vm->stack[vm->sp++] = a << (b & 0x3F); }",
        VMOpcodeBase.SHR.value: "if (vm->sp >= 2) { int64_t b = vm->stack[--vm->sp]; int64_t a = vm->stack[--vm->sp]; vm->stack[vm->sp++] = a >> (b & 0x3F); }",
        VMOpcodeBase.LT.value: "if (vm->sp >= 2) { int64_t b = vm->stack[--vm->sp]; int64_t a = vm->stack[--vm->sp]; vm->stack[vm->sp++] = (a < b) ? 1 : 0; }",
        VMOpcodeBase.LE.value: "if (vm->sp >= 2) { int64_t b = vm->stack[--vm->sp]; int64_t a = vm->stack[--vm->sp]; vm->stack[vm->sp++] = (a <= b) ? 1 : 0; }",
        VMOpcodeBase.GT.value: "if (vm->sp >= 2) { int64_t b = vm->stack[--vm->sp]; int64_t a = vm->stack[--vm->sp]; vm->stack[vm->sp++] = (a > b) ? 1 : 0; }",
        VMOpcodeBase.GE.value: "if (vm->sp >= 2) { int64_t b = vm->stack[--vm->sp]; int64_t a = vm->stack[--vm->sp]; vm->stack[vm->sp++] = (a >= b) ? 1 : 0; }",
        VMOpcodeBase.EQ.value: "if (vm->sp >= 2) { int64_t b = vm->stack[--vm->sp]; int64_t a = vm->stack[--vm->sp]; vm->stack[vm->sp++] = (a == b) ? 1 : 0; }",
        VMOpcodeBase.NE.value: "if (vm->sp >= 2) { int64_t b = vm->stack[--vm->sp]; int64_t a = vm->stack[--vm->sp]; vm->stack[vm->sp++] = (a != b) ? 1 : 0; }",
        VMOpcodeBase.JUMP.value: "vm->pc = arg;",
        VMOpcodeBase.JUMP_IF_FALSE.value: "if (vm->sp > 0) { int64_t cond = vm->stack[--vm->sp]; if (!cond) vm->pc = arg; }",
        VMOpcodeBase.JUMP_IF_TRUE.value: "if (vm->sp > 0) { int64_t cond = vm->stack[--vm->sp]; if (cond) vm->pc = arg; }",
        VMOpcodeBase.RETURN.value: "if (vm->sp > 0) return vm->stack[--vm->sp]; return 0;",
        VMOpcodeBase.DUP.value: "if (vm->sp > 0 && vm->sp < 1024) { vm->stack[vm->sp] = vm->stack[vm->sp-1]; vm->sp++; }",
        VMOpcodeBase.POP.value: "if (vm->sp > 0) vm->sp--;",
        VMOpcodeBase.SWAP.value: "if (vm->sp >= 2) { int64_t tmp = vm->stack[vm->sp-1]; vm->stack[vm->sp-1] = vm->stack[vm->sp-2]; vm->stack[vm->sp-2] = tmp; }",
        VMOpcodeBase.NOP.value: "/* no-op */",
    }
    
    for base_op, handler in op_handlers.items():
        opcode_cases += f"            case 0x{base_op:02x}: {handler} break;\n"
    
    return f'''// Complete Fix VM Runtime - Auto-generated
// Version: {__version__}
#include <stdint.h>
#include <string.h>

typedef int64_t int64;

typedef struct {{
    int64 stack[1024];
    int64 vars[256];
    int64 params[16];
    int sp;
    int pc;
    int param_count;
}} VM;

static VM _vm_instance = {{0}};

int64 {function_name}(VM* vm, const unsigned char* code, int code_len,
                      int64* input_params, int param_count) {{
    // Initialize VM state
    vm->pc = 0;
    vm->sp = 0;
    vm->param_count = param_count;
    memset(vm->vars, 0, sizeof(vm->vars));
    
    // Load parameters
    for (int i = 0; i < param_count && i < 16; i++) {{
        vm->params[i] = input_params[i];
    }}
    
    int max_cycles = 10000000;
    int cycles = 0;
    
    // CRITICAL: Bytecode format is [opcode:1][arg:4] repeated
    // Each instruction is 5 bytes total
    int num_instructions = code_len / 5;
    
    while (vm->pc < num_instructions && cycles < max_cycles) {{
        cycles++;
        
        // Read opcode (1 byte)
        int offset = vm->pc * 5;
        unsigned char opcode = code[offset];
        
        // Read argument (4 bytes, little-endian, signed)
        int32_t arg = 0;
        arg |= ((int32_t)code[offset + 1] << 0);
        arg |= ((int32_t)code[offset + 2] << 8);
        arg |= ((int32_t)code[offset + 3] << 16);
        arg |= ((int32_t)code[offset + 4] << 24);
        
        vm->pc++;
        
        // Execute instruction (NO MAPPING - use raw opcodes)
        switch (opcode) {{
{opcode_cases}
            default:
                break;
        }}
    }}
    
    return vm->sp > 0 ? vm->stack[--vm->sp] : 0;
}}

int64 execute_vm(const unsigned char* code, int code_len, int64* params, int param_count) {{
    return {function_name}(&_vm_instance, code, code_len, params, param_count);
}}
'''


class VirtualMachine:
    """VM compatible with main.py"""
    
    def __init__(self, security_level = None):
        if security_level is None:
            security_level = SecurityLevel.STANDARD
        self.security_level = security_level
        self.compiler = VMCompiler()
        self.mapping = self.compiler.mapping
        self.enabled = (security_level == SecurityLevel.PARANOID)
    
    def compile_function(self, func_def: ast.FunctionDef) -> List[Tuple[VMOpcodeBase, Optional[int]]]:
        """Compile function to bytecode"""
        return self.compiler.compile_function(func_def)
    
    def generate_c_runtime(self) -> str:
        """Generate C runtime code"""
        return generate_c_runtime(self.mapping, "vm_execute")
    
    def generate_vm_runtime(self) -> str:
        """Alias for main.py compatibility"""
        return self.generate_c_runtime()


class VMOpcode:
    """Legacy compatibility"""
    pass


__all__ = [
    'VMCompiler',
    'VirtualMachine',
    'VMOpcode',
    'generate_c_runtime',
]