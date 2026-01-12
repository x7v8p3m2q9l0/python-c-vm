import ast
import secrets
import hashlib
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

@dataclass
class MutationConfig:
    """Configuration for code mutations"""
    enable_arithmetic: bool = True
    enable_control_flow: bool = True
    enable_register_swap: bool = True
    enable_junk_code: bool = True
    mutation_strength: int = 3  # 1-5

class PolymorphicEngine:
    """
    Advanced polymorphic code engine with:
    - Arithmetic mutations (ADD <-> SUB+NEG)
    - Register/variable swapping
    - Control flow mutations
    - Junk code insertion
    - Instruction reordering
    - Equivalent instruction substitution
    """
    
    def __init__(self, config: Optional[MutationConfig] = None):
        self.config = config or MutationConfig()
        self.mutation_count = 0
    
    # ================================================================
    # ARITHMETIC MUTATIONS
    # ================================================================
    
    def mutate_arithmetic(self, expression: str) -> str:
        """
        Mutate arithmetic expressions to equivalent forms
        
        Examples:
            x + y  ->  x - (-y)
            x * 2  ->  x << 1
            x / 2  ->  x >> 1
        """
        if not self.config.enable_arithmetic:
            return expression
        
        mutations = []
        
        # ADD <-> SUB with NEG
        if '+' in expression:
            mutations.append(expression.replace('+', '- (-', 1) + ')')
        
        # SUB <-> ADD with NEG
        if '-' in expression and '- (-' not in expression:
            mutations.append(expression.replace('-', '+ (-', 1) + ')')
        
        # MUL by power of 2 <-> SHL
        if '* 2' in expression:
            mutations.append(expression.replace('* 2', '<< 1'))
        elif '* 4' in expression:
            mutations.append(expression.replace('* 4', '<< 2'))
        elif '* 8' in expression:
            mutations.append(expression.replace('* 8', '<< 3'))
        
        # DIV by power of 2 <-> SHR  
        if '/ 2' in expression:
            mutations.append(expression.replace('/ 2', '>> 1'))
        elif '/ 4' in expression:
            mutations.append(expression.replace('/ 4', '>> 2'))
        elif '/ 8' in expression:
            mutations.append(expression.replace('/ 8', '>> 3'))
        
        # XOR mutations
        # x ^ 0 = x
        # x ^ x = 0
        # x ^ (~x) = -1
        
        if mutations:
            self.mutation_count += 1
            return secrets.choice(mutations)
        
        return expression
    
    # ================================================================
    # CONSTANT MUTATIONS
    # ================================================================
    
    def mutate_constant(self, value: int) -> str:
        """
        Mutate constant into equivalent expression
        
        Examples:
            42 -> (43 - 1)
            42 -> (21 * 2)
            42 -> (84 >> 1)
            42 -> (0x2A)
            42 -> (42 ^ 17 ^ 17)
        """
        mutations = [
            # Identity mutations
            f"({value})",
            f"({value} + 0)",
            f"({value} - 0)",
            f"({value} * 1)",
            f"(~(~{value}))",
            
            # Arithmetic mutations
            f"({value + 1} - 1)",
            f"({value - 1} + 1)",
            f"({value * 2} / 2)",
            f"({value * 2} >> 1)",
            
            # XOR mutations (key ^ key = 0, 0 ^ value = value)
            f"({value} ^ {secrets.randbelow(256)} ^ {secrets.randbelow(256)})",
            
            # Hexadecimal
            f"(0x{value:X})",
        ]
        
        # Octal for small values
        if value < 512:
            mutations.append(f"(0o{value:o})")
        
        # Binary for very small values
        if value < 256:
            mutations.append(f"(0b{value:b})")
        
        self.mutation_count += 1
        return secrets.choice(mutations)
    
    # ================================================================
    # CONTROL FLOW MUTATIONS
    # ================================================================
    
    def mutate_if_statement(self, condition: str, true_block: str, false_block: str = "") -> str:
        """
        Mutate if-statement to equivalent forms
        
        Examples:
            if (x > 0) { A } else { B }
            ->
            if (!(x <= 0)) { A } else { B }
            ->
            if (x <= 0) { B } else { A }
        """
        if not self.config.enable_control_flow:
            if false_block:
                return f"if ({condition}) {{\n{true_block}\n}} else {{\n{false_block}\n}}"
            return f"if ({condition}) {{\n{true_block}\n}}"
        
        mutations = []
        
        # Original form
        if false_block:
            mutations.append(f"if ({condition}) {{\n{true_block}\n}} else {{\n{false_block}\n}}")
        else:
            mutations.append(f"if ({condition}) {{\n{true_block}\n}}")
        
        # Negated condition (swap branches)
        if false_block:
            mutations.append(f"if (!({condition})) {{\n{false_block}\n}} else {{\n{true_block}\n}}")
        
        # Ternary-like conditional jump (C)
        # More complex control flow
        
        self.mutation_count += 1
        return secrets.choice(mutations)
    
    # ================================================================
    # JUNK CODE INSERTION
    # ================================================================
    
    def generate_junk_code(self, num_lines: int = 3) -> List[str]:
        """
        Generate junk code that has no effect
        
        Examples:
            int64 _junk = 0;
            _junk = _junk + 1 - 1;
            if (_junk == 0) { }
        """
        if not self.config.enable_junk_code:
            return []
        
        junk_lines = []
        
        for i in range(num_lines):
            junk_var = f"_j{secrets.randbelow(1000)}"
            
            junk_types = [
                # Do-nothing arithmetic
                f"int64 {junk_var} = 0;",
                f"{junk_var} = {junk_var} + 1 - 1;",
                f"{junk_var} = {junk_var} ^ {junk_var};",
                f"{junk_var} = {junk_var} * 1;",
                
                # Always-false conditions
                f"if ({junk_var} > {junk_var}) {{ }}",
                f"if ({junk_var} != {junk_var}) {{ }}",
                
                # Opaque predicates (always true)
                f"if ({junk_var} == {junk_var}) {{ {junk_var} = 0; }}",
            ]
            
            junk_lines.append("    " + secrets.choice(junk_types))
        
        self.mutation_count += len(junk_lines)
        return junk_lines
    
    # ================================================================
    # REGISTER/VARIABLE SWAPPING
    # ================================================================
    
    def mutate_register_allocation(self, code: str, var_map: Dict[str, str]) -> str:
        """
        Swap variable names (register allocation mutation)
        
        Args:
            code: Code to mutate
            var_map: Mapping of old_name -> new_name
        
        Returns:
            Mutated code
        """
        if not self.config.enable_register_swap:
            return code
        
        mutated = code
        for old_name, new_name in var_map.items():
            # Replace whole words only
            import re
            mutated = re.sub(r'\b' + old_name + r'\b', new_name, mutated)
        
        self.mutation_count += 1
        return mutated
    
    # ================================================================
    # INSTRUCTION REORDERING
    # ================================================================
    
    def mutate_instruction_order(self, instructions: List[str]) -> List[str]:
        """
        Reorder independent instructions
        
        Only reorders instructions that don't have dependencies
        """
        if len(instructions) < 2:
            return instructions
        
        # Simple implementation: randomly swap adjacent independent instructions
        # More sophisticated analysis would build dependency graph
        
        result = instructions.copy()
        
        for _ in range(len(result) // 2):
            i = secrets.randbelow(len(result) - 1)
            
            # Check if instructions are independent (simplified check)
            if self._are_independent(result[i], result[i+1]):
                result[i], result[i+1] = result[i+1], result[i]
                self.mutation_count += 1
        
        return result
    
    def _are_independent(self, instr1: str, instr2: str) -> bool:
        """Check if two instructions are independent (simplified)"""
        # Very simplified - proper implementation would analyze data dependencies
        # Don't reorder if either contains function calls, returns, or control flow
        keywords = ['return', 'if', 'while', 'for', 'goto', 'call']
        
        for keyword in keywords:
            if keyword in instr1.lower() or keyword in instr2.lower():
                return False
        
        return True
    
    # ================================================================
    # OPAQUE PREDICATES
    # ================================================================
    
    def generate_opaque_predicate(self, always_true: bool = True) -> str:
        """
        Generate opaque predicate (condition with known result)
        
        Args:
            always_true: If True, generate always-true predicate, else always-false
        
        Returns:
            Condition string
        """
        if always_true:
            predicates = [
                "(x * x >= 0)",  # Always true for real numbers
                "(x == x)",
                "((x & 1) == 0 || (x & 1) == 1)",  # Bit is either 0 or 1
                "(x + 1 > x || x == INT64_MAX)",
            ]
        else:
            predicates = [
                "(x != x)",
                "(x < x)",
                "((x & 1) == 0 && (x & 1) == 1)",  # Bit can't be both
            ]
        
        # Replace 'x' with random variable name
        predicate = secrets.choice(predicates)
        var_name = f"_v{secrets.randbelow(100)}"
        return predicate.replace('x', var_name)
    
    # ================================================================
    # POLYMORPHIC GENERATION
    # ================================================================
    
    def generate_variants(self, code: str, num_variants: int = 3) -> List[str]:
        """
        Generate multiple functionally equivalent variants of code
        
        Args:
            code: Original code
            num_variants: Number of variants to generate
        
        Returns:
            List of code variants (including original)
        """
        variants = [code]
        
        for _ in range(num_variants - 1):
            variant = code
            
            # Apply random mutations
            if secrets.randbelow(2):
                # Insert junk code
                junk = self.generate_junk_code(secrets.randbelow(3) + 1)
                lines = variant.split('\n')
                insert_pos = secrets.randbelow(len(lines))
                lines[insert_pos:insert_pos] = junk
                variant = '\n'.join(lines)
            
            variants.append(variant)
        
        return variants
    
    def get_statistics(self) -> Dict[str, int]:
        """Get mutation statistics"""
        return {
            'total_mutations': self.mutation_count,
            'mutation_strength': self.config.mutation_strength
        }


__all__ = ['PolymorphicEngine', 'MutationConfig']
