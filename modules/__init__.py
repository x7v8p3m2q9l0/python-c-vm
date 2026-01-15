__version__ = "6.0.2"  # bump

# Core modules
from .base import (
    VERSION,
    MAGIC_HEADER,
    CONTAINER_VERSION,
    SecurityLevel,
    OptimizationLevel,
    CompilationError,
    TranspilationError,
    SecurityError,
    IntegrityError
)

# Type system
from .type import (
    FunctionMetadata,
    BinarySection,
    VMOpcodeBase,
    OpcodeMapping,
    CompilationConfig,
    BuildInfo,
    ProtectionLayer,
    LicenseInfo,
    TypeSystem
)

# Utilities
from .utils import (
    RandomGenerator,
    SymbolTable
)

# Data obfuscation
from .data import (
    DataObfuscator,
    PythonObfuscator
)

# Control flow obfuscation
from .control_flow import (
    OpaquePredicateGenerator,
    ControlFlowFlattener
)

# Optimization
from .optimizer import (
    Optimizer
)

from .optimizer_v2 import (
    UltimateOptimizer
)

# Hardware binding
from .hardware_binding import (
    HardwareBinding
)

# Binary container
from .binary_container import (
    BinaryContainer
)

# Security monitoring
from .security_monitor import (
    SecurityMonitor
)

from .security_v2 import (
    UltimateSecurityMonitor,
    ThreatLevel,
    SecurityEvent
)

from .memory_loader import (
    MemoryLoader
)

# Library wrapper
from .libwrapper import (
    CompiledLibrary
)

# VM Compiler
from .vm import (
    VMCompiler,
    VirtualMachine,
    VMOpcode,
    generate_c_runtime
)

try:
    from .license_system import (
        LicenseManager,
        License,
        LicenseType,
        LicenseStatus,
        LicenseError,
        MachineIDGenerator
    )
except ImportError:
    # License system dependencies not met; skip import
    pass
from .polymorphic_engine import (
    PolymorphicEngine,
    MutationConfig
)

from .decompbreak import (
    BreakerAST
)

__all__ = [
    # Version info
    'VERSION',
    '__version__',
    
    # Base classes
    'MAGIC_HEADER',
    'CONTAINER_VERSION',
    'SecurityLevel',
    'OptimizationLevel',
    'CompilationError',
    'TranspilationError',
    'SecurityError',
    'IntegrityError',
    
    # Type system
    'FunctionMetadata',
    'BinarySection',
    'VMOpcodeBase',
    'OpcodeMapping',
    'CompilationConfig',
    'BuildInfo',
    'ProtectionLayer',
    'LicenseInfo',
    'TypeSystem',
    
    # Utilities
    'RandomGenerator',
    'SymbolTable',
    
    # Obfuscation
    'DataObfuscator',
    'PythonObfuscator',
    'OpaquePredicateGenerator',
    'ControlFlowFlattener',
    
    # Optimization
    'Optimizer',
    'UltimateOptimizer',
    
    # Security
    'HardwareBinding',
    'SecurityMonitor',
    'UltimateSecurityMonitor',
    'ThreatLevel',
    'SecurityEvent',
    
    # Binary handling
    'BinaryContainer',
    'MemoryLoader',
    'CompiledLibrary',
    
    # VM Compiler
    'VMCompiler',
    'VirtualMachine',
    'VMOpcode',
    'generate_c_runtime',
    
    # License System
    # 'LicenseManager',
    # 'License',
    # 'LicenseType',
    # 'LicenseStatus',
    # 'LicenseError',
    # 'MachineIDGenerator',
    
    # Polymorphic Engine
    'PolymorphicEngine',
    'MutationConfig',

    # Decompiler Breaker
    'BreakerAST'
]
