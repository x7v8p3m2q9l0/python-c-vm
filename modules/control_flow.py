from .utils import RandomGenerator
from typing import List, Set, Dict, Optional, Tuple
import secrets
import re
import hashlib

class OpaquePredicateGenerator:    
    @staticmethod
    def always_true() -> str:
        """Generate mathematically always-true predicate"""
        c = RandomGenerator.random_int(1, 100)
        templates = [
            # Mathematical invariants
            f"(({c} * {c}) >= 0)",  # Square is always non-negative
            f"(({c} & 1) == 0 || ({c} & 1) == 1)",  # Bit is 0 or 1
            
            # Identity operations
            f"(({c} | 0) == {c})",
            f"(({c} ^ 0) == {c})",
            f"(({c} + 0) == {c})",
            f"(({c} - 0) == {c})",
            
            # Tautologies
            f"(({c} == {c}))",
            f"(({c} >= {c}))",
            f"(({c} <= {c}))",
            
            # Bitwise tautologies
            f"((~{c} | {c}) == -1)",  # Complement OR original = all bits set
            f"((~{c} ^ -1) == {c})",  # Double complement
            
            # Complex expressions
            f"((({c} + 1) > {c}) || {c} == 9223372036854775807LL)",  # Handle overflow
            f"(({c} ^ {c}) == 0)",  # XOR with self
        ]
        return secrets.choice(templates)
    
    @staticmethod
    def always_false() -> str:
        """Generate mathematically always-false predicate"""
        c = RandomGenerator.random_int(1, 100)
        templates = [
            # Contradictions
            f"(({c} != {c}))",
            f"(({c} < {c}))",
            f"(({c} > {c}))",
            
            # Bitwise contradictions
            f"(({c} & (~{c})) != 0)",  # AND with complement
            f"(({c} ^ {c}) != 0)",  # XOR with self
            
            # Arithmetic impossibilities
            f"(({c} * 0) != 0)",
            f"(({c} - {c}) != 0)",
            
            # Complex contradictions
            f"((({c} & 1) == 0) && (({c} & 1) == 1))",  # Can't be both
            f"((0) != (0))",
            f"((1) == (0))",
        ]
        return secrets.choice(templates)
    
    @staticmethod
    def contextual_true(var_name: str) -> str:
        """Generate always-true predicate using a variable"""
        templates = [
            f"(({var_name} * {var_name}) >= 0)",
            f"(({var_name} == {var_name}))",
            f"(({var_name} ^ 0) == {var_name})",
            f"((({var_name} & 1) == 0) || (({var_name} & 1) == 1))",
        ]
        return secrets.choice(templates)
    
    @staticmethod
    def random_condition() -> str:
        """Generate random opaque condition"""
        return (OpaquePredicateGenerator.always_true() 
                if secrets.randbelow(2) 
                else OpaquePredicateGenerator.always_false())


class Statement:
    """Represents a single statement or block of code"""
    
    def __init__(self, code: str, stmt_type: str = "normal"):
        self.code = code.strip()
        self.stmt_type = stmt_type  # normal, return, break, continue, conditional
        self.has_control_flow = self._detect_control_flow()
        self.declares_vars = self._extract_declarations()
        self.uses_vars = self._extract_variable_usage()
    
    def _detect_control_flow(self) -> bool:
        """Detect if statement contains control flow"""
        keywords = ['return', 'break', 'continue', 'goto']
        return any(keyword in self.code for keyword in keywords)
    
    def _extract_declarations(self) -> Set[str]:
        """Extract variable declarations"""
        declarations = set()
        # Match: type varname = ...
        # Patterns: int64 x = ..., int32 y = ..., etc.
        pattern = r'\b(int64|int32|int16|int8|uint64|uint32|uint16|uint8|double|float)\s+(\w+)\s*='
        matches = re.findall(pattern, self.code)
        for type_name, var_name in matches:
            declarations.add((type_name, var_name))
        return declarations
    
    def _extract_variable_usage(self) -> Set[str]:
        """Extract variables used (read from) in statement"""
        # This is simplified - proper implementation would use AST parsing
        # For now, extract potential variable names
        words = re.findall(r'\b[a-zA-Z_]\w*\b', self.code)
        # Filter out keywords and type names
        keywords = {
            'if', 'else', 'while', 'for', 'return', 'break', 'continue',
            'int64', 'int32', 'int16', 'int8', 'uint64', 'uint32', 'uint16', 'uint8',
            'double', 'float', 'void', 'const', 'static', 'goto', 'switch', 'case'
        }
        return set(w for w in words if w not in keywords)
    
    def remove_type_from_declaration(self) -> str:
        """Convert 'int64 x = 5' to 'x = 5'"""
        pattern = r'\b(int64|int32|int16|int8|uint64|uint32|uint16|uint8|double|float)\s+(\w+)\s*='
        return re.sub(pattern, r'\2 =', self.code)
    
    def __repr__(self):
        return f"Statement({self.stmt_type}, {len(self.code)} chars)"


class ControlFlowAnalyzer:
    """Analyzes control flow to determine if flattening is safe"""
    
    def __init__(self, statements: List[Statement]):
        self.statements = statements
    
    def can_flatten(self) -> Tuple[bool, str]:
        """
        Determine if statements can be safely flattened
        Returns: (can_flatten, reason)
        """
        
        # Check 1: Too few statements
        if len(self.statements) < 2:
            return False, "Too few statements (need at least 2)"
        
        # Check 2: Early returns (except last statement)
        for i, stmt in enumerate(self.statements[:-1]):
            if 'return' in stmt.code:
                return False, f"Early return at statement {i}"
        
        # Check 3: Break/continue (these require loop context)
        for i, stmt in enumerate(self.statements):
            if 'break' in stmt.code or 'continue' in stmt.code:
                return False, f"Break/continue at statement {i}"
        
        # Check 4: Complex nested control flow
        for i, stmt in enumerate(self.statements):
            # Count braces to detect nesting depth
            open_braces = stmt.code.count('{')
            close_braces = stmt.code.count('}')
            if open_braces > 2 or close_braces > 2:
                return False, f"Complex nesting at statement {i}"
        
        # Check 5: Goto statements
        for i, stmt in enumerate(self.statements):
            if 'goto' in stmt.code:
                return False, f"Goto at statement {i}"
        
        # Check 6: Multiple control flow paths in one statement
        for i, stmt in enumerate(self.statements):
            if stmt.code.count('if (') > 1:
                return False, f"Multiple conditionals at statement {i}"
        
        return True, "Safe to flatten"
    
    def extract_all_declarations(self) -> Dict[str, str]:
        """Extract all variable declarations"""
        declarations = {}
        for stmt in self.statements:
            for type_name, var_name in stmt.declares_vars:
                declarations[var_name] = type_name
        return declarations


class ControlFlowFlattener:
    """
    Enhanced control-flow flattening (state machine transformation)
    
    Converts sequential code into a state machine to obscure control flow.
    Example:
        x = 1;
        y = x + 2;
        return y;
    
    Becomes:
        int64 x;
        int64 y;
        int64 _state = 42;
        while (1) {
            switch (_state) {
                case 42:
                    x = 1;
                    _state = 17;
                    break;
                case 17:
                    y = x + 2;
                    _state = 93;
                    break;
                case 93:
                    return y;
                default:
                    goto _exit;
            }
        }
        _exit:;
    """
    
    def __init__(self, seed: Optional[bytes] = None):
        self.state_counter = 0
        self.seed = seed or secrets.token_bytes(16)
    
    def flatten(self, stmts: List[str], indent_level: int = 1) -> str:
        """
        Convert sequential statements to state machine
        
        Args:
            stmts: List of statement strings
            indent_level: Current indentation level
        
        Returns:
            Flattened code as string
        """
        if len(stmts) < 2:
            return "\n".join(stmts)
        
        # Parse statements
        statements = [Statement(s) for s in stmts]
        
        # Analyze control flow
        analyzer = ControlFlowAnalyzer(statements)
        can_flatten, reason = analyzer.can_flatten()
        
        if not can_flatten:
            # Cannot safely flatten - return original
            return "\n".join(stmts)
        
        # Extract all variable declarations
        all_declarations = analyzer.extract_all_declarations()
        
        # Generate state IDs (deterministic but random-looking)
        state_ids = self._generate_state_ids(len(statements))
        
        # Build flattened code
        return self._build_state_machine(
            statements, 
            state_ids, 
            all_declarations, 
            indent_level
        )
    
    def _generate_state_ids(self, count: int) -> List[int]:
        """Generate random-looking state IDs"""
        # Use hash-based generation for determinism with seed
        state_ids = []
        for i in range(count):
            hash_input = self.seed + i.to_bytes(4, 'little')
            hash_val = int(hashlib.sha256(hash_input).hexdigest()[:8], 16)
            # Keep in reasonable range
            state_id = (hash_val % 9000) + 1000  # Range: 1000-9999
            state_ids.append(state_id)
        
        # Ensure uniqueness
        while len(set(state_ids)) != len(state_ids):
            # Regenerate if collision
            for i in range(len(state_ids)):
                if state_ids.count(state_ids[i]) > 1:
                    state_ids[i] = RandomGenerator.random_int(1000, 9999)
        
        return state_ids
    
    def _build_state_machine(
        self, 
        statements: List[Statement],
        state_ids: List[int],
        declarations: Dict[str, str],
        indent_level: int
    ) -> str:
        """Build the state machine code"""
        ind = "    " * indent_level
        lines = []
        
        # Generate unique state variable name
        state_var = f"_state_{self.state_counter}"
        exit_label = f"_exit_{self.state_counter}"
        self.state_counter += 1
        
        # 1. Emit all variable declarations at function scope
        for var_name, type_name in sorted(declarations.items()):
            lines.append(f"{ind}{type_name} {var_name};")
        
        # 2. Initialize state variable
        lines.append(f"{ind}int64 {state_var} = {state_ids[0]};")
        
        # 3. Add infinite loop with always-true predicate
        loop_condition = OpaquePredicateGenerator.always_true()
        lines.append(f"{ind}while ({loop_condition}) {{")
        
        # 4. Add switch statement
        lines.append(f"{ind}    switch ({state_var}) {{")
        
        # 5. Generate case for each statement
        for i, (stmt, state_id) in enumerate(zip(statements, state_ids)):
            lines.append(f"{ind}    case {state_id}:")
            
            # Remove type declarations from statement
            stmt_code = stmt.remove_type_from_declaration()
            
            # Add statement lines
            for line in stmt_code.split('\n'):
                if line.strip():
                    lines.append(f"{ind}        {line.strip()}")
            
            # Add occasional fake branches (opaque predicates)
            if i > 0 and secrets.randbelow(3) == 0:  # 33% chance
                fake_state = RandomGenerator.random_int(10000, 99999)
                fake_condition = OpaquePredicateGenerator.always_false()
                lines.append(f"{ind}        if ({fake_condition}) {{")
                lines.append(f"{ind}            {state_var} = {fake_state};")
                lines.append(f"{ind}            break;")
                lines.append(f"{ind}        }}")
            
            # State transition
            if i < len(statements) - 1:
                # Next state
                lines.append(f"{ind}        {state_var} = {state_ids[i + 1]};")
                lines.append(f"{ind}        break;")
            else:
                # Last state - check if it's a return
                if 'return' in stmt.code:
                    # Let the return execute naturally
                    pass
                else:
                    # Exit the state machine
                    lines.append(f"{ind}        goto {exit_label};")
        
        # 6. Add fake dead states to confuse analysis
        num_fake_states = RandomGenerator.random_int(2, 5)
        fake_state_ids = [
            RandomGenerator.random_int(10000, 99999) 
            for _ in range(num_fake_states)
        ]
        
        for fake_id in fake_state_ids:
            lines.append(f"{ind}    case {fake_id}:")
            # Random fake operations
            fake_ops = [
                f"{state_var} = {state_var} ^ 0",
                f"{state_var} = {state_ids[0]}",  # Loop back
                f"/* unreachable */",
            ]
            lines.append(f"{ind}        {secrets.choice(fake_ops)};")
            lines.append(f"{ind}        break;")
        
        # 7. Default case
        lines.append(f"{ind}    default:")
        lines.append(f"{ind}        goto {exit_label};")
        
        # 8. Close switch and while
        lines.append(f"{ind}    }}")
        lines.append(f"{ind}}}")
        
        # 9. Exit label
        lines.append(f"{ind}{exit_label}:;")
        
        return "\n".join(lines)


class AdvancedControlFlowFlattener(ControlFlowFlattener):
    """
    Advanced version with additional features:
    - Variable dependency analysis
    - Instruction reordering
    - Context-sensitive opaque predicates
    """
    
    def __init__(self, seed: Optional[bytes] = None, 
                 enable_reordering: bool = True,
                 enable_bogus_states: bool = True):
        super().__init__(seed)
        self.enable_reordering = enable_reordering
        self.enable_bogus_states = enable_bogus_states
    
    def _can_reorder(self, stmt1: Statement, stmt2: Statement) -> bool:
        """
        Check if two statements can be safely reordered
        Based on data dependency analysis
        """
        # If either has control flow, cannot reorder
        if stmt1.has_control_flow or stmt2.has_control_flow:
            return False
        
        # Check for write-after-read (WAR) dependency
        # stmt1 writes to var that stmt2 reads
        vars_written_by_stmt1 = {var for _, var in stmt1.declares_vars}
        if vars_written_by_stmt1 & stmt2.uses_vars:
            return False
        
        # Check for read-after-write (RAW) dependency  
        # stmt2 writes to var that stmt1 reads
        vars_written_by_stmt2 = {var for _, var in stmt2.declares_vars}
        if vars_written_by_stmt2 & stmt1.uses_vars:
            return False
        
        # Check for write-after-write (WAW) dependency
        # Both write to same variable
        if vars_written_by_stmt1 & vars_written_by_stmt2:
            return False
        
        # No dependencies - safe to reorder
        return True
    
    def _reorder_statements(self, statements: List[Statement]) -> List[Statement]:
        """Reorder independent statements"""
        if not self.enable_reordering or len(statements) < 2:
            return statements
        
        # Simple greedy reordering
        reordered = statements.copy()
        
        for _ in range(len(reordered) // 2):
            i = RandomGenerator.random_int(0, len(reordered) - 2)
            
            if self._can_reorder(reordered[i], reordered[i + 1]):
                # Swap
                reordered[i], reordered[i + 1] = reordered[i + 1], reordered[i]
        
        return reordered
    
    def flatten(self, stmts: List[str], indent_level: int = 1) -> str:
        """Enhanced flatten with reordering"""
        if len(stmts) < 2:
            return "\n".join(stmts)
        
        # Parse statements
        statements = [Statement(s) for s in stmts]
        
        # Analyze control flow
        analyzer = ControlFlowAnalyzer(statements)
        can_flatten, reason = analyzer.can_flatten()
        
        if not can_flatten:
            return "\n".join(stmts)
        
        # Reorder if enabled
        if self.enable_reordering:
            statements = self._reorder_statements(statements)
        
        # Extract all variable declarations
        all_declarations = analyzer.extract_all_declarations()
        
        # Generate state IDs
        state_ids = self._generate_state_ids(len(statements))
        
        # Build flattened code
        return self._build_state_machine(
            statements, 
            state_ids, 
            all_declarations, 
            indent_level
        )


# Backward compatibility
__all__ = [
    'ControlFlowFlattener',
    'AdvancedControlFlowFlattener',
    'OpaquePredicateGenerator',
    'Statement',
    'ControlFlowAnalyzer'
]