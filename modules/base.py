from enum import Enum
VERSION = "5.0.0"
MAGIC_HEADER = b"PY2C"
CONTAINER_VERSION = 1

class SecurityLevel(Enum):
    """Security/obfuscation levels"""
    MINIMAL = 0      # Basic stripping only
    STANDARD = 1     # Moderate obfuscation
    AGGRESSIVE = 2   # Heavy obfuscation
    PARANOID = 3     # Maximum obfuscation


class CompilationError(Exception):
    """Compilation failed"""
    pass

class TranspilationError(Exception):
    """Transpilation failed"""
    pass

class SecurityError(Exception):
    """Security check failed"""
    pass

class IntegrityError(Exception):
    """Integrity verification failed"""
    pass