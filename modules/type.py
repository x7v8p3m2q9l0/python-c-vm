import ast
from dataclasses import dataclass
@dataclass
class FunctionMetadata:
    original_name: str
    obfuscated_name: str
    param_count: int
    index: int

@dataclass
class BinarySection:
    """Binary container section"""
    name: str
    data: bytes
    compressed: bool = False
    encrypted: bool = False


class TypeSystem:
    # Type definitions
    INT64 = "int64"
    FLOAT64 = "float64"
    STRING = "string"
    ARRAY = "array"
    
    @staticmethod
    def infer_type(node: ast.expr) -> str:
        """Infer type from AST node"""
        if isinstance(node, ast.Constant):
            if isinstance(node.value, int):
                return TypeSystem.INT64
            elif isinstance(node.value, float):
                return TypeSystem.FLOAT64
            elif isinstance(node.value, str):
                return TypeSystem.STRING
        return TypeSystem.INT64  # Default
