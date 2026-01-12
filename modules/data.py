from .utils import RandomGenerator
from .base import SecurityLevel
import secrets
class DataObfuscator:
    """Obfuscates constants and expressions"""
    
    @staticmethod
    def obfuscate_int(value: int) -> str:
        """Obfuscate integer constant"""
        if value == 0:
            variants = [
                "(0)",
                "(1 - 1)",
                "(x ^ x)",
                "(x & 0)",
            ]
            return secrets.choice(variants).replace('x', str(RandomGenerator.random_int(1, 100)))
        
        if value == 1:
            variants = [
                "(1)",
                "(0 + 1)",
                "((x / x))",
                "((x != 0) ? 1 : 0)",
            ]
            return secrets.choice(variants).replace('x', str(RandomGenerator.random_int(1, 100)))
        
        # For other values, use arithmetic identities
        offset = RandomGenerator.random_int(-50, 50)
        if offset == 0:
            offset = 1
        
        xor_key = RandomGenerator.random_int(1, 255)
        
        variants = [
            f"({value} + {offset} - {offset})",
            f"({value + offset} - {offset})",
            f"({value * 2} / 2)",
            f"(({value} ^ {xor_key}) ^ {xor_key})",  # Use same XOR key twice!
        ]
        return secrets.choice(variants)
    
    @staticmethod
    def obfuscate_expression(expr: str) -> str:
        """Add arithmetic identities to expression"""
        noise = RandomGenerator.random_int(1, 100)
        variants = [
            f"(({expr}) + {noise} - {noise})",
            f"(({expr}) * 1)",
            f"(({expr}) | 0)",
            f"(({expr}) ^ 0)",
        ]
        return secrets.choice(variants)

class PythonObfuscator:
    """Phase 7: Python-level obfuscation for loader code"""
    
    def __init__(self, security_level: SecurityLevel):
        self.security_level = security_level
    
    def obfuscate_string(self, s: str) -> str:
        """Encrypt string with runtime decryption"""
        if self.security_level.value < SecurityLevel.AGGRESSIVE.value:
            return repr(s)
        
        # XOR with random key
        key = secrets.token_bytes(16)
        encrypted = bytes(c ^ key[i % len(key)] for i, c in enumerate(s.encode()))
        
        return f"bytes([x^k[i%len(k)] for i,x in enumerate({list(encrypted)})]).decode() if (k:={list(key)}) else ''"
    
    def add_control_flow_obfuscation(self, code: str) -> str:
        """Add control-flow obfuscation to Python code"""
        # Keep minimal - Python indentation is sensitive
        if self.security_level.value < SecurityLevel.PARANOID.value:
            return code
        
        # Add some safe obfuscations that don't break indentation
        lines = code.split('\n')
        result = []
        
        for line in lines:
            result.append(line)
            # Add occasional fake operations that compile away
            if 'def ' in line and self.security_level == SecurityLevel.PARANOID:
                indent = len(line) - len(line.lstrip())
                result.append(' ' * (indent + 4) + '_ = None ')
        
        return '\n'.join(result)
    def add_fake_functions(self) -> str:
        """Generate fake functions to confuse analysis"""
        if self.security_level.value < SecurityLevel.AGGRESSIVE.value:
            return ""
        
        fake_funcs = []
        for i in range(RandomGenerator.random_int(3, 8)):
            name = f"_{RandomGenerator.random_id()}"
            fake_funcs.append(f"""
def {name}():
    '''Fake function - never called'''
    x = {RandomGenerator.random_int(1, 1000)}
    return x * {RandomGenerator.random_int(1, 100)} + {RandomGenerator.random_int(1, 100)}
""")
        
        return '\n'.join(fake_funcs)
    
    def add_namespace_pollution(self) -> str:
        """Add fake symbols to confuse analysis"""
        if self.security_level.value < SecurityLevel.PARANOID.value:
            return ""
        
        symbols = []
        for i in range(RandomGenerator.random_int(5, 15)):
            name = f"_{RandomGenerator.random_id()}"
            value = RandomGenerator.random_int(0, 10000)
            symbols.append(f"{name} = {value}")
        
        return '\n'.join(symbols)

# ENHANCED: More obfuscation variants
class EnhancedDataObfuscator(DataObfuscator):
    """Enhanced obfuscator with more variants"""
    
    @staticmethod
    def obfuscate_int_advanced(value: int) -> str:
        """Advanced integer obfuscation with more variants"""
        if value == 0:
            variants = [
                "(0)",
                "(1 - 1)",
                "(x ^ x)",
                "(x & 0)",
                "(x * 0)",
                "((x >> 31) & 0)",
            ]
            return secrets.choice(variants).replace('x', str(RandomGenerator.random_int(1, 100)))
        
        # Use multiple transformations
        offset1 = RandomGenerator.random_int(-100, 100)
        offset2 = RandomGenerator.random_int(-100, 100)
        xor_key = RandomGenerator.random_int(1, 255)
        
        variants = [
            f"({value} + {offset1} - {offset1})",
            f"(({value + offset1}) - {offset1})",
            f"({value * 2} / 2)",
            f"(({value} ^ {xor_key}) ^ {xor_key})",
            f"((~(~{value})))",
            f"(({value} << 0))",
            f"(({value} + {offset1} + {offset2}) - {offset1} - {offset2})",
        ]
        return secrets.choice(variants)
