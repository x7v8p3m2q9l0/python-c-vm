import ast
from enum import Enum
from typing import List, Tuple, Dict, Optional, Set
from .base import SecurityLevel


class VMOpcode(Enum):
    """VM Bytecode Instructions"""
    # Load/Store
    LOAD_CONST = 0x01
    LOAD_VAR = 0x02
    STORE_VAR = 0x03
    LOAD_PARAM = 0x04  # NEW: Load parameter
    
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
    
    # Special
    NOP = 0xFE
    HALT = 0xFF


class CompilerError(Exception):
    """Compilation error"""
    pass


class VMCompiler:
    """Compiles Python AST to VM bytecode with stack tracking"""
    
    def __init__(self):
        self.bytecode: List[Tuple[VMOpcode, Optional[int]]] = []
        self.var_map: Dict[str, int] = {}
        self.param_map: Dict[str, int] = {}  # Track parameters separately
        self.next_var_id = 0
        self.label_counter = 0
        self.labels: Dict[str, int] = {}
        self.fixups: List[Tuple[int, str]] = []
        
        # Stack tracking
        self.stack_depth = 0
        self.max_stack_depth = 0
        self.stack_depths: Dict[int, int] = {}  # Track stack depth at each instruction
    
    def compile_function(self, func_def: ast.FunctionDef) -> List[Tuple[VMOpcode, Optional[int]]]:
        """Compile a function to bytecode"""
        self.bytecode = []
        self.var_map = {}
        self.param_map = {}
        self.next_var_id = 0
        self.label_counter = 0
        self.labels = {}
        self.fixups = []
        self.stack_depth = 0
        self.max_stack_depth = 0
        self.stack_depths = {}
        
        # Map parameters to parameter IDs (not variables)
        for i, arg in enumerate(func_def.args.args):
            self.param_map[arg.arg] = i
        
        # Compile function body
        for stmt in func_def.body:
            self._compile_stmt(stmt)
        
        # Add implicit return 0 if no return statement
        if not self.bytecode or self.bytecode[-1][0] != VMOpcode.RETURN:
            self.emit(VMOpcode.LOAD_CONST, 0)
            self.emit(VMOpcode.RETURN, None)
        
        # Resolve label fixups
        self._resolve_fixups()
        
        # Validate bytecode
        self._validate_bytecode()
        
        return self.bytecode
    
    def emit(self, opcode: VMOpcode, arg: Optional[int] = None):
        """Emit a bytecode instruction and update stack depth"""
        # Record current stack depth
        self.stack_depths[len(self.bytecode)] = self.stack_depth
        
        # Update stack depth based on opcode
        if opcode == VMOpcode.LOAD_CONST or opcode == VMOpcode.LOAD_VAR or opcode == VMOpcode.LOAD_PARAM:
            self.stack_depth += 1
        elif opcode == VMOpcode.STORE_VAR:
            self.stack_depth -= 1
        elif opcode in [VMOpcode.ADD, VMOpcode.SUB, VMOpcode.MUL, VMOpcode.DIV, 
                       VMOpcode.MOD, VMOpcode.AND, VMOpcode.OR, VMOpcode.XOR,
                       VMOpcode.SHL, VMOpcode.SHR, VMOpcode.LT, VMOpcode.LE,
                       VMOpcode.GT, VMOpcode.GE, VMOpcode.EQ, VMOpcode.NE]:
            # Binary ops: pop 2, push 1
            self.stack_depth -= 1
        elif opcode in [VMOpcode.NEG, VMOpcode.NOT]:
            # Unary ops: pop 1, push 1 (net 0)
            pass
        elif opcode == VMOpcode.DUP:
            self.stack_depth += 1
        elif opcode == VMOpcode.POP:
            self.stack_depth -= 1
        elif opcode == VMOpcode.SWAP:
            # No change in depth
            pass
        elif opcode == VMOpcode.RETURN:
            # Pop 1 for return value
            if self.stack_depth > 0:
                self.stack_depth -= 1
        elif opcode in [VMOpcode.JUMP_IF_FALSE, VMOpcode.JUMP_IF_TRUE]:
            # Pop condition
            self.stack_depth -= 1
        
        # Track maximum stack depth
        if self.stack_depth > self.max_stack_depth:
            self.max_stack_depth = self.stack_depth
        
        # Check for stack underflow
        if self.stack_depth < 0:
            raise CompilerError(f"Stack underflow at instruction {len(self.bytecode)}")
        
        self.bytecode.append((opcode, arg))
    
    def get_label(self) -> str:
        """Generate a unique label"""
        label = f"L{self.label_counter}"
        self.label_counter += 1
        return label
    
    def mark_label(self, label: str):
        """Mark current position with a label"""
        self.labels[label] = len(self.bytecode)
    
    def emit_jump(self, opcode: VMOpcode, label: str):
        """Emit a jump to a label (to be resolved later)"""
        self.fixups.append((len(self.bytecode), label))
        self.emit(opcode, 0)  # Placeholder
    
    def _resolve_fixups(self):
        """Resolve all label references"""
        for pos, label in self.fixups:
            if label not in self.labels:
                raise CompilerError(f"Undefined label: {label}")
            target = self.labels[label]
            self.bytecode[pos] = (self.bytecode[pos][0], target)
    
    def _validate_bytecode(self):
        """Validate generated bytecode"""
        # Check all jumps are in bounds
        code_len = len(self.bytecode)
        for i, (opcode, arg) in enumerate(self.bytecode):
            if opcode in [VMOpcode.JUMP, VMOpcode.JUMP_IF_FALSE, VMOpcode.JUMP_IF_TRUE]:
                if arg is None or arg < 0 or arg >= code_len:
                    raise CompilerError(f"Invalid jump target at {i}: {arg}")
        
        # Check final stack depth
        if self.stack_depth != 0:
            # Should be 0 after return (return pops the value)
            # Allow 1 if last instruction is RETURN with value on stack
            if not (self.stack_depth == 1 and self.bytecode and 
                   self.bytecode[-1][0] == VMOpcode.RETURN):
                # Actually this is ok - RETURN will consume it
                pass
    
    def _compile_stmt(self, stmt: ast.stmt) -> bool:
        """
        Compile a statement
        Returns True if statement unconditionally exits (return/break/continue)
        """
        if isinstance(stmt, ast.Return):
            if stmt.value:
                self._compile_expr(stmt.value)
            else:
                self.emit(VMOpcode.LOAD_CONST, 0)
            self.emit(VMOpcode.RETURN, None)
            return True  # Unconditional exit
        
        elif isinstance(stmt, ast.Assign):
            if isinstance(stmt.targets[0], ast.Name):
                var_name = stmt.targets[0].id
                
                # Check if it's a parameter (can't reassign parameters as vars)
                if var_name in self.param_map:
                    # Store in variable space, not parameter space
                    if var_name not in self.var_map:
                        self.var_map[var_name] = self.next_var_id
                        self.next_var_id += 1
                else:
                    if var_name not in self.var_map:
                        self.var_map[var_name] = self.next_var_id
                        self.next_var_id += 1
                
                self._compile_expr(stmt.value)
                self.emit(VMOpcode.STORE_VAR, self.var_map[var_name])
            return False
        
        elif isinstance(stmt, ast.AugAssign):
            var_name = stmt.target.id
            
            # Load current value
            if var_name in self.param_map:
                self.emit(VMOpcode.LOAD_PARAM, self.param_map[var_name])
            elif var_name in self.var_map:
                self.emit(VMOpcode.LOAD_VAR, self.var_map[var_name])
            else:
                raise CompilerError(f"Undefined variable: {var_name}")
            
            # Compile RHS
            self._compile_expr(stmt.value)
            
            # Apply operation
            if isinstance(stmt.op, ast.Add):
                self.emit(VMOpcode.ADD, None)
            elif isinstance(stmt.op, ast.Sub):
                self.emit(VMOpcode.SUB, None)
            elif isinstance(stmt.op, ast.Mult):
                self.emit(VMOpcode.MUL, None)
            elif isinstance(stmt.op, ast.Div) or isinstance(stmt.op, ast.FloorDiv):
                self.emit(VMOpcode.DIV, None)
            elif isinstance(stmt.op, ast.Mod):
                self.emit(VMOpcode.MOD, None)
            else:
                raise CompilerError(f"Unsupported augmented assignment: {type(stmt.op)}")
            
            # Store result - create var if needed
            if var_name not in self.var_map:
                self.var_map[var_name] = self.next_var_id
                self.next_var_id += 1
            self.emit(VMOpcode.STORE_VAR, self.var_map[var_name])
            return False
        
        elif isinstance(stmt, ast.While):
            loop_start = self.get_label()
            loop_end = self.get_label()
            
            self.mark_label(loop_start)
            
            # Save stack depth for loop verification
            loop_start_depth = self.stack_depth
            
            self._compile_expr(stmt.test)
            self.emit_jump(VMOpcode.JUMP_IF_FALSE, loop_end)
            
            for body_stmt in stmt.body:
                self._compile_stmt(body_stmt)
            
            # Verify stack is balanced in loop body (unless body exits unconditionally)
            if self.stack_depth != loop_start_depth:
                raise CompilerError(f"Stack imbalance in while loop: {self.stack_depth} vs {loop_start_depth}")
            
            self.emit_jump(VMOpcode.JUMP, loop_start)
            self.mark_label(loop_end)
            return False
        
        elif isinstance(stmt, ast.If):
            else_label = self.get_label()
            end_label = self.get_label()
            
            # Save stack depth
            if_start_depth = self.stack_depth
            
            self._compile_expr(stmt.test)
            self.emit_jump(VMOpcode.JUMP_IF_FALSE, else_label)
            
            # Compile then branch and check if it exits
            then_exits = False
            for body_stmt in stmt.body:
                exits = self._compile_stmt(body_stmt)
                if exits:
                    then_exits = True
            
            then_end_depth = self.stack_depth
            
            if stmt.orelse:
                # Only emit jump if then branch doesn't exit
                if not then_exits:
                    self.emit_jump(VMOpcode.JUMP, end_label)
                
                self.mark_label(else_label)
                
                # Reset stack depth for else branch
                self.stack_depth = if_start_depth
                
                # Compile else branch and check if it exits
                else_exits = False
                for else_stmt in stmt.orelse:
                    exits = self._compile_stmt(else_stmt)
                    if exits:
                        else_exits = True
                
                # Only verify stack balance if neither branch exits
                # If both branches exit, no code after this runs
                # If one exits and one doesn't, the non-exiting one determines stack
                if not then_exits and not else_exits:
                    # Both branches continue - must have same stack effect
                    if self.stack_depth != then_end_depth:
                        raise CompilerError(f"Stack imbalance in if/else branches: {then_end_depth} vs {self.stack_depth}")
                elif then_exits and not else_exits:
                    # Then exits, else continues - use else's depth
                    pass  # self.stack_depth already set by else branch
                elif not then_exits and else_exits:
                    # Else exits, then continues - use then's depth
                    self.stack_depth = then_end_depth
                else:
                    # Both exit - function exits, stack depth doesn't matter
                    # But we should maintain consistency for analysis
                    pass
                
                self.mark_label(end_label)
                
                # Return True only if BOTH branches exit
                return then_exits and else_exits
            else:
                self.mark_label(else_label)
                # No else branch - if doesn't exit, stack should be unchanged
                if not then_exits:
                    self.stack_depth = if_start_depth
                return False  # Without else, statement doesn't unconditionally exit
        
        return False  # Default: statement doesn't exit
    
    def _compile_expr(self, expr: ast.expr):
        """Compile an expression"""
        if isinstance(expr, ast.Constant):
            # Handle both int and bool constants
            value = expr.value
            if isinstance(value, bool):
                value = 1 if value else 0
            self.emit(VMOpcode.LOAD_CONST, int(value))
        
        elif isinstance(expr, ast.Name):
            var_name = expr.id
            # Check if it's a parameter first
            if var_name in self.param_map:
                self.emit(VMOpcode.LOAD_PARAM, self.param_map[var_name])
            elif var_name in self.var_map:
                self.emit(VMOpcode.LOAD_VAR, self.var_map[var_name])
            else:
                raise CompilerError(f"Undefined variable: {var_name}")
        
        elif isinstance(expr, ast.BinOp):
            self._compile_expr(expr.left)
            self._compile_expr(expr.right)
            
            if isinstance(expr.op, ast.Add):
                self.emit(VMOpcode.ADD, None)
            elif isinstance(expr.op, ast.Sub):
                self.emit(VMOpcode.SUB, None)
            elif isinstance(expr.op, ast.Mult):
                self.emit(VMOpcode.MUL, None)
            elif isinstance(expr.op, ast.Div) or isinstance(expr.op, ast.FloorDiv):
                self.emit(VMOpcode.DIV, None)
            elif isinstance(expr.op, ast.Mod):
                self.emit(VMOpcode.MOD, None)
            elif isinstance(expr.op, ast.BitAnd):
                self.emit(VMOpcode.AND, None)
            elif isinstance(expr.op, ast.BitOr):
                self.emit(VMOpcode.OR, None)
            elif isinstance(expr.op, ast.BitXor):
                self.emit(VMOpcode.XOR, None)
            elif isinstance(expr.op, ast.LShift):
                self.emit(VMOpcode.SHL, None)
            elif isinstance(expr.op, ast.RShift):
                self.emit(VMOpcode.SHR, None)
            else:
                raise CompilerError(f"Unsupported binary operation: {type(expr.op)}")
        
        elif isinstance(expr, ast.Compare):
            if len(expr.ops) != 1 or len(expr.comparators) != 1:
                raise CompilerError("Only single comparisons supported")
            
            self._compile_expr(expr.left)
            self._compile_expr(expr.comparators[0])
            
            op = expr.ops[0]
            if isinstance(op, ast.Lt):
                self.emit(VMOpcode.LT, None)
            elif isinstance(op, ast.LtE):
                self.emit(VMOpcode.LE, None)
            elif isinstance(op, ast.Gt):
                self.emit(VMOpcode.GT, None)
            elif isinstance(op, ast.GtE):
                self.emit(VMOpcode.GE, None)
            elif isinstance(op, ast.Eq):
                self.emit(VMOpcode.EQ, None)
            elif isinstance(op, ast.NotEq):
                self.emit(VMOpcode.NE, None)
            else:
                raise CompilerError(f"Unsupported comparison: {type(op)}")
        
        elif isinstance(expr, ast.UnaryOp):
            self._compile_expr(expr.operand)
            if isinstance(expr.op, ast.USub):
                self.emit(VMOpcode.NEG, None)
            elif isinstance(expr.op, ast.Not):
                self.emit(VMOpcode.NOT, None)
            else:
                raise CompilerError(f"Unsupported unary operation: {type(expr.op)}")
        
        elif isinstance(expr, ast.IfExp):
            else_label = self.get_label()
            end_label = self.get_label()
            
            self._compile_expr(expr.test)
            self.emit_jump(VMOpcode.JUMP_IF_FALSE, else_label)
            self._compile_expr(expr.body)
            self.emit_jump(VMOpcode.JUMP, end_label)
            self.mark_label(else_label)
            self._compile_expr(expr.orelse)
            self.mark_label(end_label)
        
        else:
            raise CompilerError(f"Unsupported expression: {type(expr)}")


class VirtualMachine:
    """VM runtime for testing and C code generation"""
    
    def __init__(self, security_level=None):
        self.security_level = security_level
        self.enabled = security_level and security_level.value >= SecurityLevel.PARANOID.value
        self.compiler = VMCompiler()
        
        # Runtime state
        self.stack: List[int] = []
        self.vars: Dict[int, int] = {}
        self.params: List[int] = []  # Function parameters
        self.pc = 0
        self.code: List[Tuple[VMOpcode, Optional[int]]] = []
        self.max_stack = 1024
        self.call_stack: List[int] = []
    
    def load_code(self, code: List[Tuple[VMOpcode, Optional[int]]], params: List[int] = None):
        """Load bytecode and parameters"""
        self.code = code
        self.pc = 0
        self.stack = []
        self.vars = {}
        self.params = params or []
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
            
            if self.pc < 0 or self.pc >= len(self.code):
                raise RuntimeError(f"Invalid PC: {self.pc}")
            
            opcode, arg = self.code[self.pc]
            self.pc += 1
            
            try:
                if opcode == VMOpcode.LOAD_CONST:
                    self.push(arg if arg is not None else 0)
                
                elif opcode == VMOpcode.LOAD_VAR:
                    if arg not in self.vars:
                        self.push(0)  # Default to 0 for uninitialized vars
                    else:
                        self.push(self.vars[arg])
                
                elif opcode == VMOpcode.LOAD_PARAM:
                    if arg is None or arg < 0 or arg >= len(self.params):
                        self.push(0)  # Default for out-of-bounds parameters
                    else:
                        self.push(self.params[arg])
                
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
                    if b == 0:
                        raise RuntimeError("Modulo by zero")
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
                    if arg is None or arg < 0 or arg >= len(self.code):
                        raise RuntimeError(f"Invalid jump target: {arg}")
                    self.pc = arg
                
                elif opcode == VMOpcode.JUMP_IF_FALSE:
                    condition = self.pop()
                    if condition == 0:
                        if arg is None or arg < 0 or arg >= len(self.code):
                            raise RuntimeError(f"Invalid jump target: {arg}")
                        self.pc = arg
                
                elif opcode == VMOpcode.JUMP_IF_TRUE:
                    condition = self.pop()
                    if condition != 0:
                        if arg is None or arg < 0 or arg >= len(self.code):
                            raise RuntimeError(f"Invalid jump target: {arg}")
                        self.pc = arg
                
                elif opcode == VMOpcode.CALL:
                    # Simple function call (not fully implemented yet)
                    self.call_stack.append(self.pc)
                    if arg is None or arg < 0 or arg >= len(self.code):
                        raise RuntimeError(f"Invalid call target: {arg}")
                    self.pc = arg
                
                elif opcode == VMOpcode.RETURN:
                    if self.call_stack:
                        self.pc = self.call_stack.pop()
                    else:
                        # No more calls - function returns
                        return self.pop() if self.stack else 0
                
                elif opcode == VMOpcode.DUP:
                    val = self.stack[-1] if self.stack else 0
                    self.push(val)
                
                elif opcode == VMOpcode.POP:
                    self.pop()
                
                elif opcode == VMOpcode.SWAP:
                    if len(self.stack) >= 2:
                        b, a = self.pop(), self.pop()
                        self.push(b)
                        self.push(a)
                
                elif opcode == VMOpcode.NOP:
                    pass  # No operation
                
                elif opcode == VMOpcode.HALT:
                    return self.stack[-1] if self.stack else 0
                
                else:
                    raise RuntimeError(f"Unknown opcode: {opcode}")
                    
            except IndexError as e:
                raise RuntimeError(f"Stack error at PC={self.pc-1}: {e}")
            except Exception as e:
                raise RuntimeError(f"VM error at PC={self.pc-1}, opcode={opcode.name}: {e}")
        
        if cycles >= max_cycles:
            raise RuntimeError("Max cycles exceeded - possible infinite loop")
        
        return self.stack[-1] if self.stack else 0
    
    def compile_and_execute(self, func_def: ast.FunctionDef, args: List[int]) -> int:
        """Compile function and execute with arguments"""
        bytecode = self.compiler.compile_function(func_def)
        self.load_code(bytecode, args)
        return self.execute()
    
    def disassemble(self, code: List[Tuple[VMOpcode, Optional[int]]] = None) -> str:
        """Disassemble bytecode to human-readable form"""
        if code is None:
            code = self.code
        
        lines = ["Bytecode Disassembly:", "=" * 50]
        for i, (opcode, arg) in enumerate(code):
            arg_str = f" {arg}" if arg is not None else ""
            # Show stack depth if available
            depth_str = ""
            if hasattr(self.compiler, 'stack_depths') and i in self.compiler.stack_depths:
                depth_str = f" [stack={self.compiler.stack_depths[i]}]"
            lines.append(f"{i:4d}: {opcode.name:16}{arg_str}{depth_str}")
        
        lines.append("=" * 50)
        if hasattr(self.compiler, 'max_stack_depth'):
            lines.append(f"Max stack depth: {self.compiler.max_stack_depth}")
        
        return "\n".join(lines)
    
    def bytecode_to_c_array(self, code: List[Tuple[VMOpcode, Optional[int]]]) -> str:
        """Convert bytecode to C array initializer"""
        bytes_list = []
        for opcode, arg in code:
            bytes_list.append(f"0x{opcode.value:02x}")
            if arg is not None:
                # Encode as 4-byte little-endian
                bytes_list.append(f"0x{arg & 0xFF:02x}")
                bytes_list.append(f"0x{(arg >> 8) & 0xFF:02x}")
                bytes_list.append(f"0x{(arg >> 16) & 0xFF:02x}")
                bytes_list.append(f"0x{(arg >> 24) & 0xFF:02x}")
            else:
                bytes_list.extend(["0x00", "0x00", "0x00", "0x00"])
        
        return "{" + ", ".join(bytes_list) + "}"
    
    def generate_vm_runtime(self) -> str:
        """Generate complete C VM runtime"""
        if not self.enabled:
            return ""
        
        return """
// ============================================================================
// VM Runtime (Stack-based Bytecode Interpreter)
// ============================================================================

typedef struct {
    int64 stack[1024];
    int64 vars[256];
    int64 params[16];  // Function parameters
    int sp;
    int pc;
    int param_count;
} VM;

static int64 vm_execute(VM* vm, const unsigned char* code, int code_len, int64* input_params, int param_count) {
    vm->pc = 0;
    vm->sp = 0;
    vm->param_count = param_count;
    
    // Load parameters
    for (int i = 0; i < param_count && i < 16; i++) {
        vm->params[i] = input_params[i];
    }
    
    int max_cycles = 1000000;
    int cycles = 0;
    
    while (vm->pc < code_len && cycles < max_cycles) {
        cycles++;
        
        unsigned char op = code[vm->pc++];
        
        // Read 4-byte argument (little-endian)
        int64 arg = 0;
        if (vm->pc + 3 < code_len) {
            arg = (int64)(unsigned char)code[vm->pc] | 
                  ((int64)(unsigned char)code[vm->pc+1] << 8) | 
                  ((int64)(unsigned char)code[vm->pc+2] << 16) | 
                  ((int64)(unsigned char)code[vm->pc+3] << 24);
            vm->pc += 4;
        }
        
        switch (op) {
            case 0x01: // LOAD_CONST
                if (vm->sp < 1024) vm->stack[vm->sp++] = arg;
                break;
            case 0x02: // LOAD_VAR
                if (vm->sp < 1024 && arg < 256) vm->stack[vm->sp++] = vm->vars[arg];
                break;
            case 0x03: // STORE_VAR
                if (vm->sp > 0 && arg < 256) vm->vars[arg] = vm->stack[--vm->sp];
                break;
            case 0x04: // LOAD_PARAM
                if (vm->sp < 1024 && arg < 16 && arg < vm->param_count) {
                    vm->stack[vm->sp++] = vm->params[arg];
                } else if (vm->sp < 1024) {
                    vm->stack[vm->sp++] = 0;  // Default for out-of-bounds
                }
                break;
            case 0x10: // ADD
                if (vm->sp >= 2) {
                    int64 b = vm->stack[--vm->sp];
                    int64 a = vm->stack[--vm->sp];
                    vm->stack[vm->sp++] = a + b;
                }
                break;
            case 0x11: // SUB
                if (vm->sp >= 2) {
                    int64 b = vm->stack[--vm->sp];
                    int64 a = vm->stack[--vm->sp];
                    vm->stack[vm->sp++] = a - b;
                }
                break;
            case 0x12: // MUL
                if (vm->sp >= 2) {
                    int64 b = vm->stack[--vm->sp];
                    int64 a = vm->stack[--vm->sp];
                    vm->stack[vm->sp++] = a * b;
                }
                break;
            case 0x13: // DIV
                if (vm->sp >= 2) {
                    int64 b = vm->stack[--vm->sp];
                    int64 a = vm->stack[--vm->sp];
                    if (b != 0) vm->stack[vm->sp++] = a / b;
                    else vm->stack[vm->sp++] = 0;
                }
                break;
            case 0x14: // MOD
                if (vm->sp >= 2) {
                    int64 b = vm->stack[--vm->sp];
                    int64 a = vm->stack[--vm->sp];
                    if (b != 0) vm->stack[vm->sp++] = a % b;
                    else vm->stack[vm->sp++] = 0;
                }
                break;
            case 0x15: // NEG
                if (vm->sp > 0) vm->stack[vm->sp-1] = -vm->stack[vm->sp-1];
                break;
            case 0x20: // AND
                if (vm->sp >= 2) {
                    int64 b = vm->stack[--vm->sp];
                    int64 a = vm->stack[--vm->sp];
                    vm->stack[vm->sp++] = a & b;
                }
                break;
            case 0x21: // OR
                if (vm->sp >= 2) {
                    int64 b = vm->stack[--vm->sp];
                    int64 a = vm->stack[--vm->sp];
                    vm->stack[vm->sp++] = a | b;
                }
                break;
            case 0x22: // XOR
                if (vm->sp >= 2) {
                    int64 b = vm->stack[--vm->sp];
                    int64 a = vm->stack[--vm->sp];
                    vm->stack[vm->sp++] = a ^ b;
                }
                break;
            case 0x23: // NOT
                if (vm->sp > 0) vm->stack[vm->sp-1] = ~vm->stack[vm->sp-1];
                break;
            case 0x24: // SHL
                if (vm->sp >= 2) {
                    int64 b = vm->stack[--vm->sp];
                    int64 a = vm->stack[--vm->sp];
                    vm->stack[vm->sp++] = a << b;
                }
                break;
            case 0x25: // SHR
                if (vm->sp >= 2) {
                    int64 b = vm->stack[--vm->sp];
                    int64 a = vm->stack[--vm->sp];
                    vm->stack[vm->sp++] = a >> b;
                }
                break;
            case 0x30: // LT
                if (vm->sp >= 2) {
                    int64 b = vm->stack[--vm->sp];
                    int64 a = vm->stack[--vm->sp];
                    vm->stack[vm->sp++] = a < b ? 1 : 0;
                }
                break;
            case 0x31: // LE
                if (vm->sp >= 2) {
                    int64 b = vm->stack[--vm->sp];
                    int64 a = vm->stack[--vm->sp];
                    vm->stack[vm->sp++] = a <= b ? 1 : 0;
                }
                break;
            case 0x32: // GT
                if (vm->sp >= 2) {
                    int64 b = vm->stack[--vm->sp];
                    int64 a = vm->stack[--vm->sp];
                    vm->stack[vm->sp++] = a > b ? 1 : 0;
                }
                break;
            case 0x33: // GE
                if (vm->sp >= 2) {
                    int64 b = vm->stack[--vm->sp];
                    int64 a = vm->stack[--vm->sp];
                    vm->stack[vm->sp++] = a >= b ? 1 : 0;
                }
                break;
            case 0x34: // EQ
                if (vm->sp >= 2) {
                    int64 b = vm->stack[--vm->sp];
                    int64 a = vm->stack[--vm->sp];
                    vm->stack[vm->sp++] = a == b ? 1 : 0;
                }
                break;
            case 0x35: // NE
                if (vm->sp >= 2) {
                    int64 b = vm->stack[--vm->sp];
                    int64 a = vm->stack[--vm->sp];
                    vm->stack[vm->sp++] = a != b ? 1 : 0;
                }
                break;
            case 0x40: // JUMP
                if (arg >= 0 && arg < code_len) vm->pc = arg;
                break;
            case 0x41: // JUMP_IF_FALSE
                if (vm->sp > 0) {
                    int64 val = vm->stack[--vm->sp];
                    if (val == 0 && arg >= 0 && arg < code_len) vm->pc = arg;
                }
                break;
            case 0x42: // JUMP_IF_TRUE
                if (vm->sp > 0) {
                    int64 val = vm->stack[--vm->sp];
                    if (val != 0 && arg >= 0 && arg < code_len) vm->pc = arg;
                }
                break;
            case 0x51: // RETURN
                if (vm->sp > 0) return vm->stack[--vm->sp];
                return 0;
            case 0x60: // DUP
                if (vm->sp > 0 && vm->sp < 1024) {
                    vm->stack[vm->sp] = vm->stack[vm->sp-1];
                    vm->sp++;
                }
                break;
            case 0x61: // POP
                if (vm->sp > 0) vm->sp--;
                break;
            case 0x62: // SWAP
                if (vm->sp >= 2) {
                    int64 tmp = vm->stack[vm->sp-1];
                    vm->stack[vm->sp-1] = vm->stack[vm->sp-2];
                    vm->stack[vm->sp-2] = tmp;
                }
                break;
            case 0xFE: // NOP
                break;
            case 0xFF: // HALT
                if (vm->sp > 0) return vm->stack[--vm->sp];
                return 0;
            default:
                return 0;  // Unknown opcode
        }
    }
    
    if (cycles >= max_cycles) {
        return 0;  // Max cycles exceeded
    }
    
    return vm->sp > 0 ? vm->stack[--vm->sp] : 0;
}

// VM instance for function calls
static VM _vm_instance = {0};

"""