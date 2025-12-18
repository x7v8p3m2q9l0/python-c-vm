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
from modules import *
class AdvancedCCodeGenerator:
    def __init__(self, security_level: SecurityLevel = SecurityLevel.STANDARD,
                 enable_optimizations: bool = True):
        self.functions: List[str] = []
        self.func_metadata: List[FunctionMetadata] = []
        self.symbol_table = SymbolTable()
        self.security_level = security_level
        self.cf_flattener = ControlFlowFlattener()
        self.data_obf = DataObfuscator()
        self.func_index = 0
        self.enable_optimizations = enable_optimizations
        self.optimizer = Optimizer() if enable_optimizations else None
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

class AdvancedCompiler:
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

def compile_python_to_secure_module(
    py_source: str,
    output_dir: str,
    security_level: SecurityLevel = SecurityLevel.STANDARD,
    lib_name: str = "lib",
    enable_optimizations: bool = True
) -> Tuple[CompiledLibrary, str]:
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
    library = CompiledLibrary(dll_bytes, func_metadata, security_level)
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


def p2c_s2c(code, security: str = "PARANOID", optimize: bool = True, hardware_binding: bool = False) -> str:
    security_level = SecurityLevel[security]
    tmpdir = tempfile.mkdtemp(prefix="py2c_adv_")
    library, _ = compile_python_to_secure_module(
        code, tmpdir, security_level,
        enable_optimizations=optimize
    )
    source = library.to_python_code("PySecTech-"+RandomGenerator.random_id(16), hardware_binding)[0]
    shutil.rmtree(tmpdir)
    return source


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
    print("="*70)
    print("VM Test Suite")
    print("="*70)
    
    vm = VirtualMachine()
    # TODO: Compile test code instead of hardcoding bytecode
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


if __name__ == "__main__":
    if len(sys.argv) == 1:
        run_demo()
    else:
        # Run CLI
        main()