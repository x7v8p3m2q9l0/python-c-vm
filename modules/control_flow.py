from .utils import RandomGenerator
from typing import List, Set
import secrets
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
