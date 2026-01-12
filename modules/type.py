import ast
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional, List, Any
import secrets
import hashlib

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


# ============================================================================
# VM OPCODES
# ============================================================================

class VMOpcodeBase(Enum):
    """Base VM Bytecode Instructions"""
    # Data operations
    LOAD_CONST = 0x01
    LOAD_VAR = 0x02
    STORE_VAR = 0x03
    LOAD_PARAM = 0x04
    
    # Arithmetic operations
    ADD = 0x10
    SUB = 0x11
    MUL = 0x12
    DIV = 0x13
    MOD = 0x14
    NEG = 0x15
    
    # Bitwise operations
    AND = 0x20
    OR = 0x21
    XOR = 0x22
    NOT = 0x23
    SHL = 0x24
    SHR = 0x25
    
    # Comparison operations
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
    
    # Function operations
    CALL = 0x50
    RETURN = 0x51
    
    # Stack operations
    DUP = 0x60
    POP = 0x61
    SWAP = 0x62
    
    # Special
    NOP = 0xFE
    HALT = 0xFF


@dataclass
class OpcodeMapping:
    """Runtime opcode mapping with encryption"""
    forward: Dict[int, int] = field(default_factory=dict)
    reverse: Dict[int, int] = field(default_factory=dict)
    salt: bytes = field(default_factory=lambda: secrets.token_bytes(32))
    checksum: bytes = field(default_factory=bytes)
    
    def __post_init__(self):
        if not self.forward:
            self._generate_mapping()
        if not self.checksum:
            self.checksum = self._compute_checksum()
    
    def _generate_mapping(self):
        """Generate random opcode mapping"""
        base_opcodes = [op.value for op in VMOpcodeBase]
        random_pool = list(range(256))
        secrets.SystemRandom().shuffle(random_pool)
        
        for i, base_op in enumerate(base_opcodes):
            runtime_op = random_pool[i]
            self.forward[base_op] = runtime_op
            self.reverse[runtime_op] = base_op
    
    def _compute_checksum(self) -> bytes:
        """Compute mapping checksum"""
        data = b''
        for base_op in sorted(self.forward.keys()):
            runtime_op = self.forward[base_op]
            data += base_op.to_bytes(1, 'little')
            data += runtime_op.to_bytes(1, 'little')
        data += self.salt
        return hashlib.sha256(data).digest()
    
    def map_opcode(self, base_opcode: int) -> int:
        """Map base opcode to runtime opcode"""
        return self.forward.get(base_opcode, base_opcode)
    
    def unmap_opcode(self, runtime_opcode: int) -> int:
        """Map runtime opcode back to base opcode"""
        return self.reverse.get(runtime_opcode, runtime_opcode)


@dataclass
class CompilationConfig:
    """Configuration for compilation process"""
    security_level: str = "STANDARD"  # MINIMAL, STANDARD, AGGRESSIVE, PARANOID
    enable_vm: bool = False
    enable_control_flow_flatten: bool = True
    enable_polymorphic: bool = False
    enable_anti_debug: bool = True
    enable_jit_stub: bool = True
    enable_optimizations: bool = True
    enable_string_encryption: bool = True
    enable_constant_folding: bool = True
    enable_dead_code_elimination: bool = True
    hardware_binding: bool = False
    bind_cpu: bool = False
    bind_mac: bool = False
    expiration_date: Optional[str] = None  # ISO format: "2025-12-31"
    watermark: Optional[str] = None
    max_execution_time: Optional[int] = None  # seconds
    allowed_ips: Optional[List[str]] = None
    require_license_key: bool = False


@dataclass
class BuildInfo:
    """Build information and metadata"""
    version: str
    build_id: str
    timestamp: int
    compiler_version: str
    target_platform: str
    security_level: str
    features: Dict[str, bool]
    watermark: Optional[str] = None
    expiration: Optional[str] = None
    
    def to_dict(self) -> dict:
        return {
            'version': self.version,
            'build_id': self.build_id,
            'timestamp': self.timestamp,
            'compiler_version': self.compiler_version,
            'target_platform': self.target_platform,
            'security_level': self.security_level,
            'features': self.features,
            'watermark': self.watermark,
            'expiration': self.expiration
        }
    
    def to_c_header(self) -> str:
        """Generate C header with build info"""
        return f'''
#define WATERMARK "{self.watermark or 'PySec'}"
#define BUILD_VERSION "{self.version}"
#define BUILD_ID "{self.build_id}"
#define BUILD_TIMESTAMP {self.timestamp}
#define BUILD_PLATFORM "{self.target_platform}"
#define SECURITY_LEVEL "{self.security_level}"
'''


@dataclass
class ProtectionLayer:
    """Individual protection layer configuration"""
    name: str
    enabled: bool
    strength: int  # 0-10
    overhead: float  # Performance overhead multiplier
    description: str


@dataclass
class LicenseInfo:
    """License validation information"""
    license_key: str
    expiration_date: Optional[str] = None
    allowed_machines: Optional[List[str]] = None
    feature_flags: Dict[str, bool] = field(default_factory=dict)
    signature: Optional[bytes] = None
    
    def is_valid(self) -> bool:
        """Check if license is valid"""
        if self.expiration_date:
            from datetime import datetime
            expiry = datetime.fromisoformat(self.expiration_date)
            if datetime.now() > expiry:
                return False
        return True


class TypeSystem:
    # Type definitions
    INT64 = "int64"
    INT32 = "int32"
    INT16 = "int16"
    INT8 = "int8"
    UINT64 = "uint64"
    UINT32 = "uint32"
    UINT16 = "uint16"
    UINT8 = "uint8"
    FLOAT64 = "float64"
    FLOAT32 = "float32"
    STRING = "string"
    ARRAY = "array"
    BOOL = "bool"
    VOID = "void"
    POINTER = "pointer"
    
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
            elif isinstance(node.value, bool):
                return TypeSystem.BOOL
        elif isinstance(node, ast.List):
            return TypeSystem.ARRAY
        return TypeSystem.INT64  # Default
    
    @staticmethod
    def get_c_type(type_name: str) -> str:
        """Convert internal type to C type"""
        type_map = {
            TypeSystem.INT64: "int64_t",
            TypeSystem.INT32: "int32_t",
            TypeSystem.INT16: "int16_t",
            TypeSystem.INT8: "int8_t",
            TypeSystem.UINT64: "uint64_t",
            TypeSystem.UINT32: "uint32_t",
            TypeSystem.UINT16: "uint16_t",
            TypeSystem.UINT8: "uint8_t",
            TypeSystem.FLOAT64: "double",
            TypeSystem.FLOAT32: "float",
            TypeSystem.STRING: "char*",
            TypeSystem.ARRAY: "void*",
            TypeSystem.BOOL: "int",
            TypeSystem.VOID: "void",
            TypeSystem.POINTER: "void*"
        }
        return type_map.get(type_name, "int64_t")
    
    @staticmethod
    def get_size(type_name: str) -> int:
        """Get size in bytes for type"""
        size_map = {
            TypeSystem.INT64: 8,
            TypeSystem.INT32: 4,
            TypeSystem.INT16: 2,
            TypeSystem.INT8: 1,
            TypeSystem.UINT64: 8,
            TypeSystem.UINT32: 4,
            TypeSystem.UINT16: 2,
            TypeSystem.UINT8: 1,
            TypeSystem.FLOAT64: 8,
            TypeSystem.FLOAT32: 4,
            TypeSystem.BOOL: 1,
            TypeSystem.POINTER: 8
        }
        return size_map.get(type_name, 8)