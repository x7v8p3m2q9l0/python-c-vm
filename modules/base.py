"""
Base module - Core classes and constants
Enhanced with additional functionality
"""
from enum import Enum
from typing import Optional
import time

VERSION = "6.0.1" # bump
MAGIC_HEADER = b"PY2C"
CONTAINER_VERSION = 2

class SecurityLevel(Enum):
    """Security/obfuscation levels with performance profiles"""
    MINIMAL = 0      # Basic stripping only - Fast
    STANDARD = 1     # Moderate obfuscation - Balanced
    AGGRESSIVE = 2   # Heavy obfuscation - Slow
    PARANOID = 3     # Maximum obfuscation - Slowest
    
    def get_overhead(self) -> float:
        """Get expected performance overhead multiplier"""
        return {
            SecurityLevel.MINIMAL: 1.0,
            SecurityLevel.STANDARD: 1.1,
            SecurityLevel.AGGRESSIVE: 1.3,
            SecurityLevel.PARANOID: 1.5
        }.get(self, 1.0)
    
    def get_description(self) -> str:
        """Get human-readable description"""
        return {
            SecurityLevel.MINIMAL: "Fast compilation, basic protection",
            SecurityLevel.STANDARD: "Balanced security and performance",
            SecurityLevel.AGGRESSIVE: "Heavy obfuscation, slower execution",
            SecurityLevel.PARANOID: "Maximum protection with VM execution"
        }.get(self, "Unknown")


class CompilationError(Exception):
    """Compilation failed"""
    def __init__(self, message: str, context: Optional[str] = None):
        super().__init__(message)
        self.context = context
        self.timestamp = time.time()


class TranspilationError(Exception):
    """Transpilation failed"""
    def __init__(self, message: str, line: Optional[int] = None):
        super().__init__(message)
        self.line = line
        self.timestamp = time.time()


class SecurityError(Exception):
    """Security check failed"""
    def __init__(self, message: str, reason: Optional[str] = None):
        super().__init__(message)
        self.reason = reason
        self.timestamp = time.time()


class IntegrityError(Exception):
    """Integrity verification failed"""
    def __init__(self, message: str, expected: Optional[str] = None, actual: Optional[str] = None):
        super().__init__(message)
        self.expected = expected
        self.actual = actual
        self.timestamp = time.time()


class OptimizationLevel(Enum):
    """Optimization levels"""
    NONE = 0
    BASIC = 1
    AGGRESSIVE = 2
    MAXIMUM = 3
    def get_passes(self) -> list:
        """Get optimization passes to apply"""
        if self == OptimizationLevel.NONE:
            return []
        elif self == OptimizationLevel.BASIC:
            return ['constant_folding', 'dead_code_elimination']
        else:  # AGGRESSIVE
            return ['constant_folding', 'dead_code_elimination', 'strength_reduction', 'inlining']


__all__ = [
    'VERSION',
    'MAGIC_HEADER',
    'CONTAINER_VERSION',
    'SecurityLevel',
    'OptimizationLevel',
    'CompilationError',
    'TranspilationError',
    'SecurityError',
    'IntegrityError',
]
