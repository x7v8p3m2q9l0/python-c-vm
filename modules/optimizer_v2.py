
import ast
from .base import OptimizationLevel
from typing import List, Dict, Any, Set, Optional

class UltimateOptimizer:

    def __init__(self, optimization_level: OptimizationLevel = OptimizationLevel.AGGRESSIVE):
        """
        Args:
            optimization_level: 0=none, 1=basic, 2=aggressive, 3=maximum
        """
        self.optimization_level = optimization_level
        self.constants = {}
        self.inlined_count = 0
        self.eliminated_count = 0

    
    def constant_folding(self, expr: ast.expr) -> ast.expr:
        if isinstance(expr, ast.BinOp):
            left = self.constant_folding(expr.left)
            right = self.constant_folding(expr.right)
            
            # Both sides are constants
            if self._is_constant(left) and self._is_constant(right):
                left_val = self._get_constant_value(left)
                right_val = self._get_constant_value(right)
                
                try:
                    if isinstance(expr.op, ast.Add):
                        result = left_val + right_val
                    elif isinstance(expr.op, ast.Sub):
                        result = left_val - right_val
                    elif isinstance(expr.op, ast.Mult):
                        result = left_val * right_val
                    elif isinstance(expr.op, ast.Div):
                        if right_val != 0:
                            result = left_val // right_val  # Integer division
                        else:
                            return expr
                    elif isinstance(expr.op, ast.Mod):
                        if right_val != 0:
                            result = left_val % right_val
                        else:
                            return expr
                    elif isinstance(expr.op, ast.Pow):
                        result = left_val ** right_val
                    elif isinstance(expr.op, ast.LShift):
                        result = left_val << right_val
                    elif isinstance(expr.op, ast.RShift):
                        result = left_val >> right_val
                    elif isinstance(expr.op, ast.BitOr):
                        result = left_val | right_val
                    elif isinstance(expr.op, ast.BitXor):
                        result = left_val ^ right_val
                    elif isinstance(expr.op, ast.BitAnd):
                        result = left_val & right_val
                    else:
                        return expr
                    
                    return ast.Constant(value=result)
                except:
                    return expr
            
            # Update expression with folded children
            expr.left = left
            expr.right = right
        
        elif isinstance(expr, ast.UnaryOp):
            operand = self.constant_folding(expr.operand)
            
            if self._is_constant(operand):
                val = self._get_constant_value(operand)
                
                if isinstance(expr.op, ast.USub):
                    return ast.Constant(value=-val)
                elif isinstance(expr.op, ast.UAdd):
                    return ast.Constant(value=val)
                elif isinstance(expr.op, ast.Not):
                    return ast.Constant(value=not val)
                elif isinstance(expr.op, ast.Invert):
                    return ast.Constant(value=~val)
            
            expr.operand = operand
        
        return expr
    
    def _is_constant(self, node: ast.expr) -> bool:
        """Check if node is a constant"""
        return isinstance(node, (ast.Constant, ast.Num, ast.NameConstant))
    
    def _get_constant_value(self, node: ast.expr) -> Any:
        """Get constant value from node"""
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.Num):
            return node.n
        elif isinstance(node, ast.NameConstant):
            return node.value
        return None

    
    def dead_code_elimination(self, stmts: List[ast.stmt]) -> List[ast.stmt]:
        """Remove unreachable code"""
        result = []
        reached_return = False
        
        for stmt in stmts:
            if reached_return:
                self.eliminated_count += 1
                continue  # Skip dead code after return
            

            if isinstance(stmt, ast.If):
                test = self.constant_folding(stmt.test)
                
                if self._is_constant(test):
                    val = self._get_constant_value(test)
                    
                    if val:
                        # Condition always true - only keep if-body
                        result.extend(self.dead_code_elimination(stmt.body))
                    else:
                        # Condition always false - only keep else-body
                        if stmt.orelse:
                            result.extend(self.dead_code_elimination(stmt.orelse))
                    
                    self.eliminated_count += 1
                    continue
                
                # Recursively optimize branches
                stmt.body = self.dead_code_elimination(stmt.body)
                if stmt.orelse:
                    stmt.orelse = self.dead_code_elimination(stmt.orelse)
            
            # Check for while False
            elif isinstance(stmt, ast.While):
                test = self.constant_folding(stmt.test)
                
                if self._is_constant(test):
                    val = self._get_constant_value(test)
                    
                    if not val:
                        # Loop never executes
                        self.eliminated_count += 1
                        continue
                
                stmt.body = self.dead_code_elimination(stmt.body)
            
            # Track returns
            if isinstance(stmt, ast.Return):
                reached_return = True
            
            result.append(stmt)
        
        return result
    
    # ================================================================
    # FUNCTION INLINING
    # ================================================================
    
    def inline_functions(self, tree: ast.Module, max_inline_size: int = 10) -> ast.Module:
        """Inline small functions"""
        if self.optimization_level.value < 2:
            return tree
        
        # Build function map
        functions = {}
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                functions[node.name] = node
        
        # Inline small functions
        inliner = FunctionInliner(functions, max_inline_size)
        tree = inliner.visit(tree)
        self.inlined_count = inliner.inlined_count
        
        return tree
    
    # ================================================================
    # LOOP UNROLLING
    # ================================================================
    
    def unroll_loops(self, stmts: List[ast.stmt], max_unroll: int = 4) -> List[ast.stmt]:
        """Unroll small constant loops"""
        if self.optimization_level.value < 2:
            return stmts
        
        result = []
        
        for stmt in stmts:
            if isinstance(stmt, ast.While):
                # Try to detect simple counting loops
                unrolled = self._try_unroll_loop(stmt, max_unroll)
                if unrolled:
                    result.extend(unrolled)
                else:
                    result.append(stmt)
            else:
                result.append(stmt)
        
        return result
    
    def _try_unroll_loop(self, loop: ast.While, max_unroll: int) -> Optional[List[ast.stmt]]:
        """Attempt to unroll a simple loop"""
        # Only unroll very simple loops for safety
        # TODO: Implement more sophisticated loop analysis
        return None
    
    # ================================================================
    # STRENGTH REDUCTION
    # ================================================================
    
    def strength_reduction(self, expr: ast.expr) -> ast.expr:
        """Replace expensive operations with cheaper equivalents"""
        if isinstance(expr, ast.BinOp):
            left = self.strength_reduction(expr.left)
            right = self.strength_reduction(expr.right)
            
            # x * 2 -> x << 1
            if isinstance(expr.op, ast.Mult):
                if self._is_power_of_2(right):
                    power = self._get_power_of_2(right)
                    return ast.BinOp(
                        left=left,
                        op=ast.LShift(),
                        right=ast.Constant(value=power)
                    )
                elif self._is_power_of_2(left):
                    power = self._get_power_of_2(left)
                    return ast.BinOp(
                        left=right,
                        op=ast.LShift(),
                        right=ast.Constant(value=power)
                    )
            
            # x / 2 -> x >> 1
            elif isinstance(expr.op, ast.Div):
                if self._is_power_of_2(right):
                    power = self._get_power_of_2(right)
                    return ast.BinOp(
                        left=left,
                        op=ast.RShift(),
                        right=ast.Constant(value=power)
                    )
            
            expr.left = left
            expr.right = right
        
        return expr
    
    def _is_power_of_2(self, node: ast.expr) -> bool:
        """Check if node is a power of 2 constant"""
        if self._is_constant(node):
            val = self._get_constant_value(node)
            if isinstance(val, int) and val > 0:
                return (val & (val - 1)) == 0
        return False
    
    def _get_power_of_2(self, node: ast.expr) -> int:
        """Get the power for a power-of-2 constant"""
        val = self._get_constant_value(node)
        power = 0
        while val > 1:
            val >>= 1
            power += 1
        return power
    
    # ================================================================
    # MAIN OPTIMIZATION PASS
    # ================================================================
    
    def optimize(self, tree: ast.Module) -> ast.Module:
        """Run all optimization passes"""
        if self.optimization_level.value == 0:
            return tree
        
        # Pass 1: Constant folding
        tree = ConstantFolder().visit(tree)
        
        # Pass 2: Dead code elimination
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                node.body = self.dead_code_elimination(node.body)
        
        # Pass 3: Strength reduction
        if self.optimization_level.value >= 2:
            tree = StrengthReducer().visit(tree)
        
        # Pass 4: Function inlining
        if self.optimization_level.value >= 2:
            tree = self.inline_functions(tree)
        
        # Pass 5: Final constant folding
        tree = ConstantFolder().visit(tree)
        
        return tree
    
    def get_stats(self) -> Dict[str, int]:
        """Get optimization statistics"""
        return {
            'inlined_functions': self.inlined_count,
            'eliminated_statements': self.eliminated_count,
            'optimization_level': self.optimization_level.name
        }


# ================================================================
# AST VISITORS FOR OPTIMIZATION PASSES
# ================================================================

class ConstantFolder(ast.NodeTransformer):
    """AST visitor for constant folding"""
    
    def visit_BinOp(self, node):
        self.generic_visit(node)
        
        optimizer = UltimateOptimizer()
        return optimizer.constant_folding(node)
    
    def visit_UnaryOp(self, node):
        self.generic_visit(node)
        
        optimizer = UltimateOptimizer()
        return optimizer.constant_folding(node)


class StrengthReducer(ast.NodeTransformer):
    """AST visitor for strength reduction"""
    
    def visit_BinOp(self, node):
        self.generic_visit(node)
        
        optimizer = UltimateOptimizer()
        return optimizer.strength_reduction(node)

class FunctionInliner(ast.NodeTransformer):
    """AST visitor for function inlining"""
    
    def __init__(self, functions: Dict[str, ast.FunctionDef], max_size: int):
        self.functions = functions
        self.max_size = max_size
        self.inlined_count = 0
    
    def visit_Call(self, node):
        self.generic_visit(node)
        
        # Only inline simple function calls
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            
            if func_name in self.functions:
                func_def = self.functions[func_name]
                
                # Check if function is small enough
                if len(func_def.body) <= self.max_size:
                    # TODO: Implement actual inlining logic
                    # This is complex and requires variable renaming
                    # For now, just count potential inlines
                    self.inlined_count += 1
        
        return node



# Backward compatibility
class Optimizer:
    """Legacy optimizer for backward compatibility"""
    
    def __init__(self):
        self.impl = UltimateOptimizer(optimization_level=OptimizationLevel.AGGRESSIVE)
    
    def constant_folding(self, expr: ast.expr) -> ast.expr:
        return self.impl.constant_folding(expr)
    
    def dead_code_elimination(self, stmts: List[ast.stmt]) -> List[ast.stmt]:
        return self.impl.dead_code_elimination(stmts)


__all__ = ['UltimateOptimizer', 'Optimizer']
