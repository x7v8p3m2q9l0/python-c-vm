import ast
from typing import List
class Optimizer:
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
