"""
Polymorphic Code Engine - Maximum Chaos Edition

Extreme unpredictability and irreversibility while maintaining functionality.
Each mutation is unique and cannot be traced back to the original.
"""

import secrets
import hashlib
import re
import ast
import copy
import random
import time
from typing import List, Dict, Tuple, Optional, Set, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod


# Seed randomness with multiple entropy sources
def _init_chaos():
    """Initialize maximum chaos"""
    entropy = int(time.time() * 1000000) ^ secrets.randbits(256)
    random.seed(entropy)

_init_chaos()


class Language(Enum):
    """Supported programming languages"""
    PYTHON = "python"
    C = "c"
    CPP = "cpp"


class MutationStrength(Enum):
    """Mutation strength levels"""
    LIGHT = 1
    MODERATE = 2
    HEAVY = 3
    EXTREME = 4
    METAMORPHIC = 5
    CHAOS = 6  # New: Maximum unpredictability


@dataclass
class MutationConfig:
    """Configuration for polymorphic mutations"""
    language: Language = Language.PYTHON
    
    # Enable mutation types
    enable_arithmetic: bool = True
    enable_control_flow: bool = True
    enable_register_swap: bool = True
    enable_junk_code: bool = True
    enable_instruction_reorder: bool = True
    enable_constant_folding: bool = True
    enable_dead_code: bool = True
    enable_opaque_predicates: bool = True
    enable_string_encoding: bool = True
    enable_variable_renaming: bool = True
    enable_control_flow_flattening: bool = True
    enable_expression_splitting: bool = True
    enable_identity_functions: bool = True
    
    # Mutation parameters
    mutation_strength: MutationStrength = MutationStrength.HEAVY
    junk_code_ratio: float = 0.3
    max_mutation_depth: int = 5
    chaos_factor: float = 0.8  # 0.0 = predictable, 1.0 = maximum chaos
    
    # Irreversibility features
    enable_hash_based_names: bool = True
    enable_multi_stage_encoding: bool = True
    enable_expression_trees: bool = True
    enable_fake_branches: bool = True
    
    # Advanced features
    preserve_semantics: bool = True
    randomize_on_each_build: bool = True
    maintain_debug_info: bool = False
    
    def get_junk_lines_count(self, total_lines: int) -> int:
        """Calculate junk lines with chaos"""
        base = int(total_lines * self.junk_code_ratio)
        if base <= 0:
            return 0
        if self.chaos_factor > 0.5:
            chaos_amount = max(1, int(base * self.chaos_factor))
            chaos = secrets.randbelow(chaos_amount)
            return base + chaos
        return base


# ============================================================================
# ADVANCED CONSTANT OBFUSCATION - IRREVERSIBLE
# ============================================================================

class ChaosConstantObfuscator:
    """
    Generate extremely complex constant expressions that are
    mathematically correct but practically irreversible
    """
    
    @staticmethod
    def generate_chaos_constant(value: int, chaos_level: int = 3) -> str:
        """Generate deeply nested, irreversible constant expression"""
        
        if value == 0:
            return ChaosConstantObfuscator._chaos_zero(chaos_level)
        elif value == 1:
            return ChaosConstantObfuscator._chaos_one(chaos_level)
        
        # For chaos_level 1-2, use only reliable strategies
        if chaos_level <= 2:
            strategies = [
                ChaosConstantObfuscator._strategy_arithmetic_tree,
            ]
        else:
            # Multi-layer obfuscation strategies
            strategies = [
                ChaosConstantObfuscator._strategy_arithmetic_tree,
                ChaosConstantObfuscator._strategy_xor_chain,
                ChaosConstantObfuscator._strategy_bit_manipulation,
                ChaosConstantObfuscator._strategy_mixed_operations,
            ]
        
        strategy = secrets.choice(strategies)
        return strategy(value, chaos_level)
    
    @staticmethod
    def _chaos_zero(chaos_level: int) -> str:
        """Generate complex zero"""
        if chaos_level <= 1:
            return secrets.choice([
                "(0)",
                "(1 - 1)",
                "(x & 0)".replace('x', str(secrets.randbelow(100))),
            ])
        
        # Multi-stage zero generation
        x = secrets.randbelow(1000) + 1
        y = secrets.randbelow(1000) + 1
        
        options = [
            f"({x} - {x})",
            f"({x} * 0)",
            f"({x} & 0)",
            f"(({x} ^ {x}) & ({y} ^ {y}))",
            f"((({x} + {y}) - {x}) - {y})",
            f"(({x} << {chaos_level}) - ({x} << {chaos_level}))",
            f"(~{x} & {x})",  # Always 0
        ]
        
        return secrets.choice(options)
    
    @staticmethod
    def _chaos_one(chaos_level: int) -> str:
        """Generate complex one"""
        if chaos_level <= 1:
            return secrets.choice([
                "(1)",
                "(2 - 1)",
                "(2 >> 1)",
            ])
        
        x = secrets.randbelow(100) + 1
        y = secrets.randbelow(100) + 1
        
        options = [
            f"({x} // {x})",
            f"(({x} + 1) - {x})",
            f"(({x} * 2) // ({x} * 2) if {x} != 0 else 1)",
            f"((~0) & 1)",
            f"(({x} ^ {y}) // ({x} ^ {y}) if ({x} ^ {y}) != 0 else 1)",
        ]
        
        return secrets.choice(options)
    
    @staticmethod
    def _strategy_arithmetic_tree(value: int, depth: int) -> str:
        """Build deep arithmetic expression tree"""
        if depth <= 0 or value < -1000 or value > 1000:
            return str(value)
        
        # Simple correct strategies
        if value == 0:
            return "(0)"
        
        max_offset = max(1, min(abs(value), 50))
        offset = secrets.randbelow(max_offset) + 1
        
        strategies = [
            f"({value + offset} - {offset})",
            f"({value - offset} + {offset})",
        ]
        
        if value > 0 and offset > 0:
            strategies.append(f"(({value * offset}) // {offset})")
        
        return secrets.choice(strategies)
    
    @staticmethod
    def _strategy_xor_chain(value: int, depth: int) -> str:
        """Build XOR chain that resolves to value"""
        if depth <= 0 or value < 0 or value > 1000:
            return str(value)
        
        # Generate random XOR chain
        num_keys = min(depth, 3)  # Limit to 3 keys max
        keys = [secrets.randbelow(0xFFFF) for _ in range(num_keys)]
        
        # Calculate: value ^ k1 ^ k2 ^ ... ^ kn ^ k1 ^ k2 ^ ... ^ kn = value
        result = value
        for key in keys:
            result ^= key
        
        # Build expression with proper parentheses
        expr = str(result)
        for key in keys:
            expr = f"({expr} ^ {key})"
        for key in keys:
            expr = f"({expr} ^ {key})"
        
        return expr
    
    @staticmethod
    def _strategy_bit_manipulation(value: int, depth: int) -> str:
        """Use bit manipulation to create value"""
        if depth <= 0 or value < 0 or value > 1000:
            return str(value)
        
        # Decompose into bit operations
        bits = []
        for i in range(16):
            if value & (1 << i):
                bits.append(i)
        
        if not bits:
            return "(0)"
        
        # Create expression using bit shifts with parentheses
        parts = [f"(1 << {bit})" for bit in bits]
        
        # Build with proper parentheses
        if len(parts) == 1:
            return parts[0]
        
        result = parts[0]
        for part in parts[1:]:
            result = f"({result} | {part})"
        
        return result
    
    @staticmethod
    def _strategy_mixed_operations(value: int, depth: int) -> str:
        """Mix multiple operations"""
        if depth <= 0:
            return str(value)
        
        # Create complex nested expression
        x = secrets.randbelow(100) + 1
        y = secrets.randbelow(100) + 1
        z = secrets.randbelow(100) + 1
        
        # Generate expression that equals value
        templates = [
            f"((({value} + {x}) * {y}) // {y} - {x})",
            f"(({value} ^ {x}) ^ {x})",
            f"((({value} << 2) + ({value} << 1)) // 6)",  # value * 6 // 6
            f"(({value} | {x}) - ({x} & ~{value}))",
            f"((({value} + {x} + {y}) - {x}) - {y})",
        ]
        
        return secrets.choice(templates)


class VariableNameChaosGenerator:
    """
    Generate extremely obfuscated variable names that are
    unique and irreversible
    """
    
    # Use visually similar characters to create confusion
    CHAOS_CHARS = {
        'level1': 'abcdefghijklmnopqrstuvwxyz',
        'level2': 'Il1O0',  # Confusing characters
        'level3': 'Il1O0oO0Il1',  # Maximum confusion
    }
    
    @staticmethod
    def generate_chaos_name(original: str, chaos_level: int = 2) -> str:
        """Generate irreversible obfuscated name"""
        
        # Hash-based naming for maximum irreversibility
        hash_input = f"{original}_{secrets.token_hex(8)}_{time.time()}"
        name_hash = hashlib.sha256(hash_input.encode()).hexdigest()
        
        if chaos_level >= 3:
            # Level 3: Maximum confusion with look-alike characters
            charset = VariableNameChaosGenerator.CHAOS_CHARS['level3']
            name_len = secrets.randbelow(8) + 8  # 8-16 chars
            name = ''.join(secrets.choice(charset) for _ in range(name_len))
        elif chaos_level >= 2:
            # Level 2: Confusing but shorter
            charset = VariableNameChaosGenerator.CHAOS_CHARS['level2']
            name_len = secrets.randbelow(4) + 6  # 6-10 chars
            name = ''.join(secrets.choice(charset) for _ in range(name_len))
        else:
            # Level 1: Hash-based
            name = f"_v{name_hash[:8]}"
        
        # Ensure it starts with letter or underscore
        if name and not (name[0].isalpha() or name[0] == '_'):
            name = '_' + name
        
        return name
    
    @staticmethod
    def generate_name_pool(count: int, chaos_level: int = 2) -> List[str]:
        """Generate pool of unique obfuscated names"""
        names = set()
        while len(names) < count:
            name = VariableNameChaosGenerator.generate_chaos_name(
                f"var_{len(names)}", chaos_level
            )
            names.add(name)
        return list(names)


class ExpressionSplitter:
    """
    Split simple expressions into complex multi-step operations
    that are functionally equivalent but irreversible
    """
    
    @staticmethod
    def split_expression(expr: str, target_var: str, depth: int = 2) -> List[str]:
        """Split expression into multiple statements"""
        if depth <= 0:
            return [f"{target_var} = {expr}"]
        
        # Create intermediate variables
        intermediates = VariableNameChaosGenerator.generate_name_pool(depth, chaos_level=2)
        
        statements = []
        
        # First intermediate: partial computation
        statements.append(f"{intermediates[0]} = {expr}")
        
        # Add identity operations
        for i in range(1, depth):
            prev = intermediates[i-1]
            curr = intermediates[i]
            
            # Identity transformations
            ops = [
                f"{curr} = {prev} + 0",
                f"{curr} = {prev} * 1",
                f"{curr} = {prev} ^ 0",
                f"{curr} = {prev} | 0",
                f"{curr} = ({prev} << 1) >> 1",
            ]
            
            statements.append(secrets.choice(ops))
        
        # Final assignment
        statements.append(f"{target_var} = {intermediates[-1]}")
        
        return statements


class ControlFlowFlattener:
    """
    Flatten control flow into state machines that are
    extremely difficult to reconstruct
    """
    
    @staticmethod
    def flatten_if_else(condition: str, true_block: str, false_block: str) -> str:
        """Convert if-else to flattened control flow"""
        
        # Generate random state IDs
        state_true = secrets.randbelow(1000) + 1
        state_false = secrets.randbelow(1000) + 1
        state_end = secrets.randbelow(1000) + 1
        
        # Use hash-based state names for irreversibility
        state_var = VariableNameChaosGenerator.generate_chaos_name("state", 2)
        
        flattened = f"""
{state_var} = {state_true} if ({condition}) else {state_false}
while True:
    if {state_var} == {state_true}:
{PolymorphicEngine._indent_block(true_block, 8)}
        {state_var} = {state_end}
    elif {state_var} == {state_false}:
{PolymorphicEngine._indent_block(false_block, 8)}
        {state_var} = {state_end}
    elif {state_var} == {state_end}:
        break
"""
        return flattened


class StringEncoder:
    """
    Multi-stage string encoding that is practically irreversible
    """
    
    @staticmethod
    def encode_string(s: str, stages: int = 3) -> str:
        """Apply multiple encoding stages"""
        
        if not s:
            return '""'
        
        import base64
        
        # Stage 1: Base64
        encoded = base64.b64encode(s.encode()).decode()
        
        if stages >= 2:
            # Stage 2: XOR with random key
            key = secrets.randbelow(255) + 1
            xor_encoded = ''.join(chr(ord(c) ^ key) for c in encoded)
            encoded = base64.b64encode(xor_encoded.encode()).decode()
            
            # Build decode expression
            expr = f"base64.b64decode('{encoded}').decode()"
            expr = f"''.join(chr(ord(c) ^ {key}) for c in {expr})"
            expr = f"base64.b64decode({expr}.encode()).decode()"
            
            return expr
        
        return f"base64.b64decode('{encoded}').decode()"


# ============================================================================
# ENHANCED PYTHON AST TRANSFORMERS
# ============================================================================

class ChaosPythonConstantTransformer(ast.NodeTransformer):
    """Transform constants with maximum chaos"""
    
    def __init__(self, config: MutationConfig):
        self.config = config
        self.mutations = 0
    
    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        """Transform constant with chaos"""
        if isinstance(node.value, int):
            chaos_level = min(5, int(self.config.chaos_factor * 5) + 1)
            expr_str = ChaosConstantObfuscator.generate_chaos_constant(
                node.value, chaos_level
            )
            
            # Parse the expression back to AST
            try:
                expr_node = ast.parse(expr_str, mode='eval').body
                self.mutations += 1
                return expr_node
            except:
                return node
        
        elif isinstance(node.value, str) and self.config.enable_multi_stage_encoding:
            stages = min(3, int(self.config.chaos_factor * 3) + 1)
            expr_str = StringEncoder.encode_string(node.value, stages)
            
            try:
                expr_node = ast.parse(expr_str, mode='eval').body
                self.mutations += 1
                return expr_node
            except:
                return node
        
        return node


class ChaosPythonVariableRenamer(ast.NodeTransformer):
    """Rename variables with maximum obfuscation"""
    
    def __init__(self, config: MutationConfig):
        self.config = config
        self.name_map: Dict[str, str] = {}
        self.protected = {
            'self', 'cls', 'True', 'False', 'None', 
            '__init__', '__name__', '__main__',
            'base64',  # Protect module names
        }
        self.chaos_level = min(3, int(config.chaos_factor * 3) + 1)
        self.function_names = set()  # Track function names to NOT rename them
    
    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        """Visit function definition - don't rename the function itself"""
        # DON'T rename the function name itself
        if not node.name.startswith('__'):
            self.function_names.add(node.name)
        
        # DO rename parameters
        for arg in node.args.args:
            if arg.arg not in self.protected:
                self._ensure_mapping(arg.arg)
                arg.arg = self.name_map[arg.arg]
        
        # Visit body
        node.body = [self.visit(stmt) for stmt in node.body]
        return node
    
    def visit_Name(self, node: ast.Name) -> ast.Name:
        """Rename variable references"""
        # Don't rename function names, protected names, or dunder names
        if (node.id not in self.protected and 
            not node.id.startswith('__') and
            node.id not in self.function_names):
            self._ensure_mapping(node.id)
            node.id = self.name_map[node.id]
        return node
    
    def _ensure_mapping(self, name: str):
        """Ensure variable has mapping"""
        if name not in self.name_map:
            self.name_map[name] = VariableNameChaosGenerator.generate_chaos_name(
                name, self.chaos_level
            )


class ChaosPythonArithmeticTransformer(ast.NodeTransformer):
    """Transform arithmetic with unpredictable mutations"""
    
    def __init__(self, config: MutationConfig):
        self.config = config
    
    def visit_BinOp(self, node: ast.BinOp) -> ast.AST:
        """Transform binary operations"""
        node.left = self.visit(node.left)
        node.right = self.visit(node.right)
        
        if secrets.randbelow(100) < int(self.config.chaos_factor * 100):
            if isinstance(node.op, ast.Add):
                # x + y => x - (-y) or (x ^ y) + 2 * (x & y)
                if secrets.randbelow(2):
                    return ast.BinOp(
                        left=node.left,
                        op=ast.Sub(),
                        right=ast.UnaryOp(op=ast.USub(), operand=node.right)
                    )
                else:
                    # x + y = (x ^ y) + 2 * (x & y)  - carry lookahead addition
                    xor_part = ast.BinOp(left=node.left, op=ast.BitXor(), right=node.right)
                    and_part = ast.BinOp(left=node.left, op=ast.BitAnd(), right=node.right)
                    mul_part = ast.BinOp(left=ast.Constant(2), op=ast.Mult(), right=and_part)
                    return ast.BinOp(left=xor_part, op=ast.Add(), right=mul_part)
            
            elif isinstance(node.op, ast.Mult):
                # x * 2^n => x << n
                if isinstance(node.right, ast.Constant) and isinstance(node.right.value, int):
                    val = node.right.value
                    if val > 0 and (val & (val - 1)) == 0:
                        shift = val.bit_length() - 1
                        return ast.BinOp(
                            left=node.left,
                            op=ast.LShift(),
                            right=ast.Constant(shift)
                        )
        
        return node


# ============================================================================
# CHAOS POLYMORPHIC ENGINE
# ============================================================================

class PolymorphicEngine:
    """
    Maximum chaos polymorphic engine with extreme unpredictability
    and irreversibility while maintaining functional correctness
    """
    
    def __init__(self, config: Optional[MutationConfig] = None):
        self.config = config or MutationConfig(mutation_strength=MutationStrength.CHAOS)
        self.stats = {
            'arithmetic_mutations': 0,
            'constant_mutations': 0,
            'control_flow_mutations': 0,
            'junk_code_lines': 0,
            'dead_code_blocks': 0,
            'instructions_reordered': 0,
            'variables_renamed': 0,
            'strings_encoded': 0,
            'expressions_split': 0,
            'control_flows_flattened': 0,
        }
        
        # Re-seed chaos on each instantiation
        _init_chaos()
    
    def mutate_code(self, code: str, language: Optional[Language] = None) -> str:
        """
        Apply maximum chaos mutations
        
        Args:
            code: Source code
            language: Language (auto-detected if None)
        
        Returns:
            Heavily mutated code that is functionally equivalent
        """
        if language is None:
            language = self._detect_language(code)
        
        if language == Language.PYTHON:
            return self._mutate_python_chaos(code)
        elif language in (Language.C, Language.CPP):
            return self._mutate_c_chaos(code)
        
        return code
    
    def _detect_language(self, code: str) -> Language:
        """Detect language"""
        if re.search(r'\bdef\s+\w+\s*\(', code) or 'import ' in code:
            return Language.PYTHON
        if '#include' in code or re.search(r'\bint\s+\w+\s*\(', code):
            if '::' in code or 'class ' in code:
                return Language.CPP
            return Language.C
        return self.config.language
    
    def _mutate_python_chaos(self, code: str) -> str:
        """Apply maximum chaos to Python code"""
        try:
            tree = ast.parse(code)
            
            # Apply transformation passes - but only once for reliability
            passes = [
                ChaosPythonConstantTransformer,
                ChaosPythonVariableRenamer,
                ChaosPythonArithmeticTransformer,
            ]
            
            # Apply each pass ONCE for reliability
            for transformer_class in passes:
                transformer = transformer_class(self.config)
                tree = transformer.visit(tree)
                
                if hasattr(transformer, 'mutations'):
                    self.stats['constant_mutations'] += transformer.mutations
                if hasattr(transformer, 'name_map'):
                    self.stats['variables_renamed'] += len(transformer.name_map)
            
            # Generate mutated code
            mutated = ast.unparse(tree)
            
            # Add chaos layers
            if self.config.enable_junk_code:
                mutated = self._add_python_chaos_junk(mutated)
            
            return mutated
            
        except Exception as e:
            print(f"Chaos mutation failed: {e}, using enhanced fallback")
            return self._mutate_python_chaos_fallback(code)
    
    def _mutate_python_chaos_fallback(self, code: str) -> str:
        """Chaos fallback using text manipulation"""
        lines = code.split('\n')
        mutated_lines = []
        
        for line in lines:
            # Mutate constants with chaos
            if re.search(r'\b\d+\b', line) and '"' not in line and "'" not in line:
                def replace_num(match):
                    num = int(match.group(0))
                    chaos_level = min(3, int(self.config.chaos_factor * 3) + 1)
                    return ChaosConstantObfuscator.generate_chaos_constant(num, chaos_level)
                
                line = re.sub(r'\b\d+\b', replace_num, line)
            
            mutated_lines.append(line)
        
        return '\n'.join(mutated_lines)
    
    def _add_python_chaos_junk(self, code: str) -> str:
        """Add unpredictable junk code"""
        lines = code.split('\n')
        junk_count = self.config.get_junk_lines_count(len(lines))
        
        for _ in range(junk_count):
            if not lines:
                break
            
            pos = secrets.randbelow(len(lines))
            indent = len(lines[pos]) - len(lines[pos].lstrip()) if pos < len(lines) else 0
            
            # Generate complex junk
            junk_var = VariableNameChaosGenerator.generate_chaos_name("junk", 2)
            junk_val = ChaosConstantObfuscator.generate_chaos_constant(
                secrets.randbelow(1000), 
                secrets.randbelow(3) + 1
            )
            
            junk_lines = [
                ' ' * indent + f"{junk_var} = {junk_val}",
            ]
            
            # Add identity operation
            if secrets.randbelow(2):
                identity_ops = [
                    f"{junk_var} = {junk_var} ^ 0",
                    f"{junk_var} = {junk_var} | 0",
                    f"{junk_var} = {junk_var} & -1",
                    f"{junk_var} = ({junk_var} << 1) >> 1",
                ]
                junk_lines.append(' ' * indent + secrets.choice(identity_ops))
            
            lines[pos:pos] = junk_lines
            self.stats['junk_code_lines'] += len(junk_lines)
        
        # Add dead code blocks
        if self.config.enable_dead_code:
            for _ in range(secrets.randbelow(3) + 1):
                pos = secrets.randbelow(len(lines))
                indent = len(lines[pos]) - len(lines[pos].lstrip()) if pos < len(lines) else 0
                
                # Always-false condition
                false_cond = ChaosConstantObfuscator._chaos_zero(2)
                dead_var = VariableNameChaosGenerator.generate_chaos_name("dead", 2)
                dead_val = secrets.randbelow(1000)
                
                dead_lines = [
                    ' ' * indent + f"if {false_cond}:",
                    ' ' * (indent + 4) + f"{dead_var} = {dead_val}",
                ]
                
                lines[pos:pos] = dead_lines
                self.stats['dead_code_blocks'] += 1
        
        return '\n'.join(lines)
    
    def _split_python_expressions(self, code: str) -> str:
        """Split expressions into multiple statements"""
        # This is complex and would need full AST analysis
        # For now, return as-is
        self.stats['expressions_split'] = 0
        return code
    
    def _flatten_python_control_flow(self, code: str) -> str:
        """Flatten control flow (simplified)"""
        # Full implementation would need comprehensive AST analysis
        # For now, return as-is
        self.stats['control_flows_flattened'] = 0
        return code
    
    def _mutate_c_chaos(self, code: str) -> str:
        """Apply chaos to C code"""
        lines = code.split('\n')
        mutated_lines = []
        
        for line in lines:
            # Skip preprocessor
            if line.strip().startswith('#'):
                mutated_lines.append(line)
                continue
            
            # Mutate constants
            if re.search(r'\b0[xX][0-9a-fA-F]+\b|\b\d+\b', line):
                def replace_num(match):
                    num_str = match.group(0)
                    try:
                        if num_str.startswith('0x') or num_str.startswith('0X'):
                            num = int(num_str, 16)
                        else:
                            num = int(num_str)
                        
                        chaos_level = min(3, int(self.config.chaos_factor * 3) + 1)
                        mutated = ChaosConstantObfuscator.generate_chaos_constant(num, chaos_level)
                        self.stats['constant_mutations'] += 1
                        return mutated
                    except:
                        return num_str
                
                line = re.sub(r'\b0[xX][0-9a-fA-F]+\b|\b\d+\b', replace_num, line)
            
            mutated_lines.append(line)
        
        # Add junk code
        if self.config.enable_junk_code:
            junk_count = self.config.get_junk_lines_count(len(mutated_lines))
            
            for _ in range(junk_count):
                pos = secrets.randbelow(len(mutated_lines))
                indent = len(mutated_lines[pos]) - len(mutated_lines[pos].lstrip())
                
                junk_var = VariableNameChaosGenerator.generate_chaos_name("junk", 2)
                junk_val = secrets.randbelow(1000)
                
                junk_line = ' ' * indent + f"int {junk_var} = {junk_val};"
                mutated_lines.insert(pos, junk_line)
                self.stats['junk_code_lines'] += 1
        
        return '\n'.join(mutated_lines)
    
    def generate_variants(self, code: str, num_variants: int = 3,
                         language: Optional[Language] = None) -> List[str]:
        """Generate extremely different variants"""
        variants = []
        
        for i in range(num_variants):
            # CRITICAL: Reset chaos with new entropy each time
            import time
            entropy = int(time.time() * 1000000) ^ secrets.randbits(256) ^ (i * 12345)
            random.seed(entropy)
            
            self.stats = {k: 0 for k in self.stats}
            
            # Reduce rounds for speed - 1-2 rounds is enough for uniqueness
            rounds = 1 if self.config.chaos_factor < 0.5 else 2
            
            variant = code
            for round_num in range(rounds):
                # Vary chaos factor per round
                old_chaos = self.config.chaos_factor
                self.config.chaos_factor = min(1.0, old_chaos + random.random() * 0.3)
                
                variant = self.mutate_code(variant, language)
                
                self.config.chaos_factor = old_chaos
            
            variants.append(variant)
        
        return variants
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get mutation statistics"""
        return {
            **self.stats,
            'total_mutations': sum(self.stats.values()),
            'mutation_strength': self.config.mutation_strength.name,
            'chaos_factor': self.config.chaos_factor,
            'language': self.config.language.name,
        }
    
    def reset_statistics(self):
        """Reset statistics"""
        self.stats = {k: 0 for k in self.stats}
    
    @staticmethod
    def _indent_block(block: str, spaces: int) -> str:
        """Indent a code block"""
        indent = ' ' * spaces
        return '\n'.join(indent + line if line.strip() else line 
                        for line in block.split('\n'))

def create_chaos_config(language: Language = Language.PYTHON, chaos_factor: float = 0.8) -> MutationConfig:
    """Create maximum chaos configuration"""
    return MutationConfig(
        language=language,
        mutation_strength=MutationStrength.CHAOS,
        junk_code_ratio=0.4,
        max_mutation_depth=5,
        chaos_factor=chaos_factor,
        enable_hash_based_names=True,
        enable_multi_stage_encoding=True,
        enable_expression_trees=True,
        enable_fake_branches=True,
        enable_control_flow_flattening=True,
        enable_expression_splitting=True,
        enable_identity_functions=True,
    )


def create_heavy_config(language: Language = Language.PYTHON) -> MutationConfig:
    """Create heavy configuration with good chaos"""
    return MutationConfig(
        language=language,
        mutation_strength=MutationStrength.HEAVY,
        junk_code_ratio=0.3,
        max_mutation_depth=3,
        chaos_factor=0.6,
        enable_hash_based_names=True,
        enable_multi_stage_encoding=True,
    )


def create_moderate_config(language: Language = Language.PYTHON) -> MutationConfig:
    """Create moderate configuration"""
    return MutationConfig(
        language=language,
        mutation_strength=MutationStrength.MODERATE,
        junk_code_ratio=0.2,
        max_mutation_depth=2,
        chaos_factor=0.4,
    )


def create_light_config(language: Language = Language.PYTHON) -> MutationConfig:
    """Create light configuration"""
    return MutationConfig(
        language=language,
        mutation_strength=MutationStrength.LIGHT,
        junk_code_ratio=0.1,
        max_mutation_depth=1,
        chaos_factor=0.2,
        enable_dead_code=False,
    )


__all__ = [
    'PolymorphicEngine',

    'MutationConfig',
    'MutationStrength',
    'Language',
    'create_chaos_config',
    'create_heavy_config',
    'create_moderate_config',
    'create_light_config',
    'ChaosConstantObfuscator',
    'VariableNameChaosGenerator',
]